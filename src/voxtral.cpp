#include "voxtral.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <chrono>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <cinttypes>
#include <string>

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#define VOXTRAL_ENC_DIM         1280
#define VOXTRAL_ENC_HEADS       32
#define VOXTRAL_ENC_HEAD_DIM    64
#define VOXTRAL_ENC_LAYERS      32
#define VOXTRAL_DEC_DIM         3072
#define VOXTRAL_DEC_HEADS       32
#define VOXTRAL_DEC_HEAD_DIM    128
#define VOXTRAL_DEC_LAYERS      26

#undef VOXTRAL_VOCAB_SIZE
#undef VOXTRAL_TOKEN_BOS
#undef VOXTRAL_TOKEN_EOS
#undef VOXTRAL_TOKEN_STREAMING_PAD

#define VOXTRAL_VOCAB_SIZE              131072
#define VOXTRAL_TOKEN_BOS               1
#define VOXTRAL_TOKEN_EOS               2
#define VOXTRAL_TOKEN_STREAMING_PAD     32

#define VOXTRAL_HOP_LENGTH      160
#define VOXTRAL_WINDOW_SIZE     400
#define VOXTRAL_N_FFT           400
#define VOXTRAL_N_FREQ          201
#define VOXTRAL_NUM_MEL_BINS    128

static constexpr float VOXTRAL_PI = 3.1415926535f;

struct voxtral_encoder_layer {
    ggml_tensor * attn_norm_weight;
    ggml_tensor * attn_q_weight; ggml_tensor * attn_q_bias;
    ggml_tensor * attn_k_weight;
    ggml_tensor * attn_v_weight; ggml_tensor * attn_v_bias;
    ggml_tensor * attn_o_weight; ggml_tensor * attn_o_bias;
    ggml_tensor * ffn_norm_weight;
    ggml_tensor * ffn_w1_weight;
    ggml_tensor * ffn_w2_weight; ggml_tensor * ffn_w2_bias;
    ggml_tensor * ffn_w3_weight;
};

struct voxtral_decoder_layer {
    ggml_tensor * attn_norm_weight;
    ggml_tensor * attn_q_weight;
    ggml_tensor * attn_k_weight;
    ggml_tensor * attn_v_weight;
    ggml_tensor * attn_o_weight;
    ggml_tensor * ffn_norm_weight;
    ggml_tensor * ffn_w1_weight;
    ggml_tensor * ffn_w2_weight;
    ggml_tensor * ffn_w3_weight;
    ggml_tensor * ada0_weight;
    ggml_tensor * ada2_weight;
};

struct voxtral_model {
    gguf_context * gguf_ctx; ggml_context * ctx_gguf;
    ggml_backend_t backend_weights; ggml_backend_buffer_t buf_weights;
    bool weights_on_gpu; voxtral_gpu_backend gpu_type;
    ggml_tensor * enc_conv0_weight; ggml_tensor * enc_conv0_bias;
    ggml_tensor * enc_conv1_weight; ggml_tensor * enc_conv1_bias;
    ggml_tensor * enc_norm_weight;
    std::vector<voxtral_encoder_layer> enc_layers;
    ggml_tensor * adapter_0_weight; ggml_tensor * adapter_2_weight;
    ggml_tensor * tok_embeddings_weight;
    ggml_tensor * dec_norm_weight;
    std::vector<voxtral_decoder_layer> dec_layers;
    ggml_tensor * mel_filters;
    std::set<int32_t> tokenizer_special_ranks;
    int32_t tokenizer_num_special_tokens = 1000;
    std::vector<std::string> tokenizer_vocab_b64;
    mutable std::map<int32_t, std::string> tokenizer_bytes_cache;
};

struct voxtral_context {
    voxtral_model * model; voxtral_log_level log_level; voxtral_log_callback logger;
    ggml_backend_t backend; ggml_backend_t backend_cpu;
    voxtral_gpu_backend gpu_type;
    ggml_backend_sched_t sched_encoder; ggml_backend_sched_t sched_adapter;
    ggml_backend_sched_t sched_dec_pre; ggml_backend_sched_t sched_dec_step;
    ggml_context * ctx_persistent; ggml_backend_buffer_t buf_persistent;
    ggml_tensor * kv_self_k; ggml_tensor * kv_self_v;
    ggml_tensor * encoder_output; ggml_tensor * decoder_memory;
    ggml_tensor * decoder_logits; ggml_tensor * encoder_chunk_output;
    std::vector<float> hann_window; std::vector<float> mel_filters_cpu;
    std::vector<float> time_emb_cpu;
    int32_t n_threads = 4; int32_t enc_seq_used = 0; int32_t kv_used = 0;
    int32_t max_parallel_streams = 1;
};

struct voxtral_stream {
    voxtral_context * ctx; int32_t slot_id; std::vector<float> audio_buf;
    int32_t samples_processed = 0; std::vector<int32_t> all_tokens;
    int32_t tokens_reported = 0; int32_t last_token = VOXTRAL_TOKEN_STREAMING_PAD;
    int32_t dec_position = 0; int32_t kv_used = 0;
    int32_t enc_tokens_total = 0; int32_t enc_kv_used = 0;
    int32_t dec_positions_total = 0; int32_t consecutive_pad = 0;
    bool seen_text = false; bool prefilled = false;
};

ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name);
ggml_tensor * get_tensor(ggml_context * ctx, const char * name);
void clear_kv_cache_slotted(voxtral_context * ctx, int32_t slot_id);
void kv_cache_shift_left_slotted(voxtral_context * ctx, int32_t slot_id, int32_t shift);
std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens);
void compute_mel_spectrogram(const float * audio, int32_t n_samples, const float * mel_filters, const float * hann_window, float * mel_out, int32_t * out_n_frames);
ggml_cgraph * build_encoder_graph(voxtral_context * ctx, ggml_context * gctx, const float * mel, int32_t frames, int32_t * out_len);
ggml_cgraph * build_adapter_graph(voxtral_context * ctx, ggml_context * gctx);
ggml_cgraph * build_adapter_graph_slice(voxtral_context * ctx, ggml_context * gctx, int32_t slot_id, int32_t e_off, int32_t e_count);
ggml_cgraph * build_decoder_prefill_graph(voxtral_context * ctx, ggml_context * gctx, int32_t slot_id, int32_t n_tokens);
ggml_cgraph * build_decoder_step_graph(voxtral_context * ctx, ggml_context * gctx, int32_t slot_id, int32_t pos, int32_t a_pos);
ggml_cgraph * build_decoder_step_batched_graph(voxtral_context * ctx, ggml_context * gctx, const int32_t * kv_offsets);
bool run_encoder_chunk(voxtral_context * ctx, const float * mel, int32_t frames, int32_t rope_off, int32_t * out_len);
bool run_adapter(voxtral_context * ctx);
bool run_decoder_prefill(voxtral_context * ctx, int32_t slot_id, const int32_t * tokens, int32_t n, float * out);
bool run_decoder_step(voxtral_context * ctx, int32_t slot_id, int32_t token, int32_t pos, int32_t a_pos, int32_t kv_used, float * out);
bool run_decoder_step_batched(voxtral_context * ctx, int32_t n, const int32_t * slot_ids, const int32_t * tokens, const int32_t * positions, const int32_t * kv_used, const float ** embs, float * out);
void voxtral_stream_reset(voxtral_stream * s);
bool stream_process_audio_to_encoder(voxtral_stream * s, const float * a, int32_t n);
bool stream_decode_available(voxtral_stream * s, std::string & text, bool early = false);
void compute_time_embedding(std::vector<float> & out, float t, int32_t dim);
double elapsed_ms(const std::chrono::steady_clock::time_point & t0) {
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name) {
    return ggml_graph_get_tensor(gf, name);
}

ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) fprintf(stderr, "voxtral: tensor '%s' not found\n", name);
    return t;
}

void clear_kv_cache_slotted(voxtral_context * ctx, int32_t slot_id) {
    if (!ctx->kv_self_k) return;
    const size_t slot_off = (size_t)slot_id * ctx->kv_self_k->nb[3];
    ggml_backend_tensor_memset(ctx->kv_self_k, 0, slot_off, ctx->kv_self_k->nb[3]);
    ggml_backend_tensor_memset(ctx->kv_self_v, 0, slot_off, ctx->kv_self_v->nb[3]);
}

void kv_cache_shift_left_slotted(voxtral_context * ctx, int32_t slot_id, int32_t shift) {
    const size_t rb = ctx->kv_self_k->nb[1], ls = ctx->kv_self_k->nb[2], soff = (size_t)slot_id * ctx->kv_self_k->nb[3];
    for (int l = 0; l < 26; l++) {
        size_t off = soff + (size_t)l * ls; std::vector<uint8_t> k(ls), v(ls);
        ggml_backend_tensor_get(ctx->kv_self_k, k.data(), off, ls);
        ggml_backend_tensor_get(ctx->kv_self_v, v.data(), off, ls);
        memmove(k.data(), k.data() + (size_t)shift * rb, (1000 - shift) * rb);
        memmove(v.data(), v.data() + (size_t)shift * rb, (1000 - shift) * rb);
        memset(k.data() + (1000 - shift) * rb, 0, (size_t)shift * rb);
        memset(v.data() + (1000 - shift) * rb, 0, (size_t)shift * rb);
        ggml_backend_tensor_set(ctx->kv_self_k, k.data(), off, ls);
        ggml_backend_tensor_set(ctx->kv_self_v, v.data(), off, ls);
    }
}

static std::string base64_decode(const std::string & in) {
    static const std::string b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; std::vector<int> T(256, -1);
    for (int i=0; i<64; i++) T[b64[i]] = i;
    int val = 0, valb = -8;
    for (uint8_t c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c]; valb += 6;
        if (valb >= 0) { out.push_back(char((val >> valb) & 0xFF)); valb -= 8; }
    }
    return out;
}

std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens) {
    std::string out;
    for (int32_t t : tokens) {
        if (t < model.tokenizer_num_special_tokens || model.tokenizer_special_ranks.count(t)) continue;
        int32_t vocab_id = t - model.tokenizer_num_special_tokens;
        if (vocab_id >= 0 && vocab_id < (int32_t)model.tokenizer_vocab_b64.size()) {
            if (model.tokenizer_bytes_cache.count(t)) { out += model.tokenizer_bytes_cache[t]; }
            else { std::string d = base64_decode(model.tokenizer_vocab_b64[vocab_id]); model.tokenizer_bytes_cache[t] = d; out += d; }
        }
    }
    return out;
}

struct stft_plan {
    int32_t n_fft = 400; int32_t n_bins = 201;
    std::vector<float> cos_table; std::vector<float> sin_table;
};

static const stft_plan & get_stft_plan() {
    static stft_plan plan = []() {
        stft_plan p; p.cos_table.resize(201 * 400); p.sin_table.resize(201 * 400);
        for (int k=0; k<201; k++) {
            for (int n=0; n<400; n++) {
                float a = 2.f*VOXTRAL_PI*k*n/400.f;
                p.cos_table[k*400+n] = cosf(a); p.sin_table[k*400+n] = sinf(a);
            }
        }
        return p;
    }();
    return plan;
}

static inline int32_t reflect_index(int32_t idx, int32_t len) {
    if (len <= 1) return 0;
    while (idx < 0 || idx >= len) { if (idx < 0) idx = -idx; else idx = 2*len-2-idx; }
    return idx;
}

void compute_mel_spectrogram(const float * audio, int32_t n_samples, const float * mel_filters, const float * hann_window, float * mel_out, int32_t * out_n_frames) {
    const int32_t n_frames = n_samples / 160; *out_n_frames = n_frames;
    if (n_frames <= 0) return;
    const stft_plan & plan = get_stft_plan();
    std::vector<float> centered(n_samples + 400);
    for (int i=0; i<n_samples+400; i++) { int src = i - 200; centered[i] = audio[reflect_index(src, n_samples)]; }
    std::vector<float> windowed(400), power(201), accum(128);
    for (int f=0; f<n_frames; f++) {
        const float * frame_ptr = centered.data() + f*160;
        for (int i=0; i<400; i++) windowed[i] = frame_ptr[i] * hann_window[i];
        for (int k=0; k<201; k++) {
            float re=0, im=0; for (int i=0; i<400; i++) { re += windowed[i]*plan.cos_table[k*400+i]; im -= windowed[i]*plan.sin_table[k*400+i]; }
            power[k] = re*re + im*im;
        }
        std::fill(accum.begin(), accum.end(), 0.f);
        for (int k=0; k<201; k++) {
            const float * w = mel_filters + k*128; float pk = power[k];
            for (int m=0; m<128; m++) accum[m] += w[m]*pk;
        }
        for (int m=0; m<128; m++) {
            float v = std::max(accum[m], 1e-10f); v = log10f(v);
            v = std::max(v, -6.5f); mel_out[m*n_frames+f] = (v+4.f)/4.f;
        }
    }
}

void compute_time_embedding(std::vector<float> & out, float t, int32_t dim) {
    out.resize(dim); const int32_t half = dim / 2;
    for (int32_t i = 0; i < half; i++) {
        const float angle = t * expf(-logf(10000.0f) * (float)i / (float)half);
        out[i] = cosf(angle); out[i + half] = sinf(angle);
    }
}

bool load_wav_file(const std::string & path, std::vector<float> & audio_out) {
    std::ifstream fin(path, std::ios::binary); if (!fin) return false;
    char header[44]; if (fin.read(header, 44).gcount() < 44) return false;
    uint32_t data_size = *(uint32_t*)(header + 40);
    audio_out.resize(data_size / 2);
    std::vector<int16_t> tmp(audio_out.size());
    if (fin.read((char*)tmp.data(), data_size).gcount() < data_size) return false;
    for (size_t i=0; i<tmp.size(); i++) audio_out[i] = tmp[i] / 32768.0f;
    return true;
}
static ggml_tensor * build_decoder_layer(voxtral_context * ctx, ggml_context * gctx, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * time_emb, int32_t layer_idx, int32_t slot_id, int32_t n_tokens, int32_t kv_offset, ggml_tensor * attn_mask) {
    auto & L = ctx->model->dec_layers[layer_idx];
    const size_t layer_off = (size_t)slot_id * ctx->kv_self_k->nb[3] + (size_t)layer_idx * ctx->kv_self_k->nb[2];
    ggml_tensor * res = x, * xn = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), L.attn_norm_weight);
    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, xn), * k = ggml_mul_mat(gctx, L.attn_k_weight, xn), * v = ggml_mul_mat(gctx, L.attn_v_weight, xn);
    q = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, q, 128, 32, n_tokens), positions, nullptr, 128, 0, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
    k = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, k, 128, 8, n_tokens), positions, nullptr, 128, 0, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_cont(gctx, ggml_reshape_2d(gctx, k, 1024, n_tokens)), ggml_view_2d(gctx, ctx->kv_self_k, 1024, n_tokens, ctx->kv_self_k->nb[1], layer_off + (size_t)kv_offset * ctx->kv_self_k->nb[1])));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_cont(gctx, ggml_reshape_2d(gctx, v, 1024, n_tokens)), ggml_view_2d(gctx, ctx->kv_self_v, 1024, n_tokens, ctx->kv_self_v->nb[1], layer_off + (size_t)kv_offset * ctx->kv_self_v->nb[1])));
    const int32_t n_kv = kv_offset + n_tokens;
    ggml_tensor * kf = ggml_view_2d(gctx, ctx->kv_self_k, 1024, n_kv, ctx->kv_self_k->nb[1], layer_off), * vf = ggml_view_2d(gctx, ctx->kv_self_v, 1024, n_kv, ctx->kv_self_v->nb[1], layer_off);
    x = ggml_flash_attn_ext(gctx, ggml_permute(gctx, q, 0, 2, 1, 3), ggml_permute(gctx, ggml_reshape_3d(gctx, kf, 128, 8, n_kv), 0, 2, 1, 3), ggml_permute(gctx, ggml_reshape_3d(gctx, vf, 128, 8, n_kv), 0, 2, 1, 3), (attn_mask?ggml_cast(gctx,attn_mask,GGML_TYPE_F16):nullptr), 1.f/sqrtf(128.f), 0.f, 0.f);
    x = ggml_add(gctx, res, ggml_mul_mat(gctx, L.attn_o_weight, ggml_reshape_2d(gctx, ggml_cont(gctx, x), 4096, n_tokens)));
    res = x; xn = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), L.ffn_norm_weight);
    { ggml_tensor * ah = ggml_mul_mat(gctx, L.ada2_weight, ggml_gelu_erf(gctx, ggml_mul_mat(gctx, L.ada0_weight, time_emb))); xn = ggml_add(gctx, xn, ggml_mul(gctx, xn, ah)); }
    return ggml_add(gctx, res, ggml_mul_mat(gctx, L.ffn_w2_weight, ggml_mul(gctx, ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_w1_weight, xn)), ggml_mul_mat(gctx, L.ffn_w3_weight, xn))));
}

static ggml_tensor * build_decoder_layer_batched(voxtral_context * ctx, ggml_context * gctx, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * time_emb, int32_t layer_idx, const int32_t * kv_offsets, ggml_tensor * msk) {
    auto & L = ctx->model->dec_layers[layer_idx];
    const int32_t N = ctx->max_parallel_streams;
    ggml_tensor * res = x, * xn = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), L.attn_norm_weight);
    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, xn), * k = ggml_mul_mat(gctx, L.attn_k_weight, xn), * v = ggml_mul_mat(gctx, L.attn_v_weight, xn);
    q = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, q, 128, 32, N), positions, nullptr, 128, 0, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
    k = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, k, 128, 8, N), positions, nullptr, 128, 0, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
    for (int b = 0; b < N; b++) {
        size_t soff = (size_t)b*ctx->kv_self_k->nb[3] + (size_t)layer_idx*ctx->kv_self_k->nb[2] + (size_t)kv_offsets[b]*ctx->kv_self_k->nb[1];
        ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_view_1d(gctx, ggml_cont(gctx, ggml_reshape_2d(gctx, k, 1024, N)), 1024, (size_t)b*1024*4), ggml_view_1d(gctx, ctx->kv_self_k, 1024, soff)));
        ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_view_1d(gctx, ggml_cont(gctx, ggml_reshape_2d(gctx, v, 1024, N)), 1024, (size_t)b*1024*4), ggml_view_1d(gctx, ctx->kv_self_v, 1024, soff)));
    }
    int32_t mk = (int32_t)msk->ne[0];
    size_t layer_off = (size_t)layer_idx * ctx->kv_self_k->nb[2];
    ggml_tensor * ka = ggml_view_4d(gctx, ctx->kv_self_k, 128, 8, mk, N, 128*ctx->kv_self_k->nb[0], ctx->kv_self_k->nb[1], ctx->kv_self_k->nb[3], layer_off);
    ggml_tensor * va = ggml_view_4d(gctx, ctx->kv_self_v, 128, 8, mk, N, 128*ctx->kv_self_v->nb[0], ctx->kv_self_v->nb[1], ctx->kv_self_v->nb[3], layer_off);
    ggml_tensor * q4 = ggml_reshape_4d(gctx, q, 128, 1, 32, N);
    x = ggml_flash_attn_ext(gctx, q4, ggml_permute(gctx, ka, 0, 2, 1, 3), ggml_permute(gctx, va, 0, 2, 1, 3), ggml_cast(gctx, msk, GGML_TYPE_F16), 1.f/sqrtf(128.f), 0.f, 0.f);
    x = ggml_add(gctx, res, ggml_mul_mat(gctx, L.attn_o_weight, ggml_reshape_2d(gctx, ggml_cont(gctx, x), 4096, N)));
    res = x; xn = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), L.ffn_norm_weight);
    { ggml_tensor * ah = ggml_mul_mat(gctx, L.ada2_weight, ggml_gelu_erf(gctx, ggml_mul_mat(gctx, L.ada0_weight, time_emb))); xn = ggml_add(gctx, xn, ggml_mul(gctx, xn, ah)); }
    return ggml_add(gctx, res, ggml_mul_mat(gctx, L.ffn_w2_weight, ggml_mul(gctx, ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_w1_weight, xn)), ggml_mul_mat(gctx, L.ffn_w3_weight, xn))));
}

ggml_cgraph * build_decoder_prefill_graph(voxtral_context * ctx, ggml_context * gctx, int32_t slot_id, int32_t n_tokens) {
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens); ggml_set_name(t, "token_ids"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, t, ctx->backend);
    ggml_tensor * p = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens); ggml_set_name(p, "positions"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, p, ctx->backend);
    ggml_tensor * tm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 3072); ggml_set_name(tm, "time_emb"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, tm, ctx->backend);
    ggml_tensor * x = ggml_add(gctx, ggml_get_rows(gctx, ctx->model->tok_embeddings_weight, t), ggml_view_2d(gctx, ctx->decoder_memory, 3072, n_tokens, ctx->decoder_memory->nb[1], (size_t)slot_id * 1000 * 3072 * 4));
    ggml_tensor * m = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens); ggml_set_name(m, "causal_mask"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, m, ctx->backend);
    for (int i=0; i<26; i++) x = build_decoder_layer(ctx, gctx, gf, x, p, tm, i, slot_id, n_tokens, 0, m);
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), ctx->model->dec_norm_weight);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_mul_mat(gctx, ctx->model->tok_embeddings_weight, ggml_view_1d(gctx, x, 3072, (n_tokens-1)*x->nb[1])), ggml_view_1d(gctx, ctx->decoder_logits, VOXTRAL_VOCAB_SIZE, (size_t)slot_id * VOXTRAL_VOCAB_SIZE * 4)));
    return gf;
}

ggml_cgraph * build_decoder_step_graph(voxtral_context * ctx, ggml_context * gctx, int32_t slot_id, int32_t pos, int32_t a_pos) {
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1); ggml_set_name(t, "token_id"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, t, ctx->backend);
    ggml_tensor * p = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1); ggml_set_name(p, "position"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, p, ctx->backend);
    ggml_tensor * tm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 3072); ggml_set_name(tm, "time_emb"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, tm, ctx->backend);
    ggml_tensor * x = ggml_add(gctx, ggml_get_rows(gctx, ctx->model->tok_embeddings_weight, t), ggml_view_2d(gctx, ctx->decoder_memory, 3072, 1, ctx->decoder_memory->nb[1], (size_t)slot_id * 1000 * 3072 * 4 + (size_t)a_pos * 3072 * 4));
    for (int i=0; i<26; i++) x = build_decoder_layer(ctx, gctx, gf, x, p, tm, i, slot_id, 1, ctx->kv_used, nullptr);
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), ctx->model->dec_norm_weight);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_mul_mat(gctx, ctx->model->tok_embeddings_weight, ggml_reshape_1d(gctx, x, 3072)), ggml_view_1d(gctx, ctx->decoder_logits, VOXTRAL_VOCAB_SIZE, (size_t)slot_id * VOXTRAL_VOCAB_SIZE * 4)));
    return gf;
}

ggml_cgraph * build_decoder_step_batched_graph(voxtral_context * ctx, ggml_context * gctx, const int32_t * kv_offsets) {
    const int32_t N = ctx->max_parallel_streams;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, N); ggml_set_name(t, "token_ids"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, t, ctx->backend);
    ggml_tensor * p = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, N); ggml_set_name(p, "positions"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, p, ctx->backend);
    ggml_tensor * tm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 3072); ggml_set_name(tm, "time_emb"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, tm, ctx->backend);
    ggml_tensor * a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 3072, N); ggml_set_name(a, "audio_emb"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, a, ctx->backend);
    int32_t mk = 0; for (int b=0; b<N; b++) mk = std::max(mk, kv_offsets[b]+1);
    ggml_tensor * msk = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, mk, 1, 1, N); ggml_set_name(msk, "msk"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, msk, ctx->backend);
    ggml_tensor * x = ggml_add(gctx, ggml_get_rows(gctx, ctx->model->tok_embeddings_weight, t), a);
    for (int i=0; i<26; i++) x = build_decoder_layer_batched(ctx, gctx, gf, x, p, tm, i, kv_offsets, msk);
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), ctx->model->dec_norm_weight);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_mul_mat(gctx, ctx->model->tok_embeddings_weight, x), ggml_view_2d(gctx, ctx->decoder_logits, VOXTRAL_VOCAB_SIZE, N, ctx->decoder_logits->nb[1], 0)));
    return gf;
}

struct causal_conv1d_dims { int32_t pad_left=0; int32_t pad_right=0; int32_t out_len=0; };
static causal_conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t k_sz, int32_t stride) {
    causal_conv1d_dims out; const int32_t p_tot = k_sz - stride;
    const float n_f = (float)(in_len - k_sz + p_tot)/stride + 1.f;
    const int32_t target = (int32_t)std::ceil(n_f - 1)*stride + (k_sz - p_tot);
    out.pad_left = p_tot; out.pad_right = std::max(0, target - in_len);
    out.out_len = (in_len + out.pad_left + out.pad_right - k_sz)/stride + 1;
    return out;
}

static ggml_tensor * causal_conv1d_graph(ggml_context * ctx, ggml_tensor * x, int32_t in_len, ggml_tensor * weight, ggml_tensor * bias, int32_t out_ch, int32_t k_sz, int32_t stride, int32_t & o_len) {
    auto dims = compute_causal_conv1d_dims(in_len, k_sz, stride);
    ggml_tensor * xp = ggml_pad_ext(ctx, x, dims.pad_left, dims.pad_right, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = ggml_conv_1d(ctx, weight, xp, stride, 0, 1);
    if (bias) y = ggml_add(ctx, y, ggml_reshape_3d(ctx, bias, 1, out_ch, 1));
    o_len = dims.out_len; return y;
}

ggml_cgraph * build_encoder_graph(voxtral_context * ctx, ggml_context * gctx, const float * mel, int32_t frames, int32_t * out_len) {
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);
    ggml_tensor * in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, frames, 128, 1);
    ggml_set_name(in, "mel_input"); ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, in, ctx->backend);
    int32_t c0l=0, c1l=0;
    ggml_tensor * x = ggml_gelu_erf(gctx, causal_conv1d_graph(gctx, in, frames, ctx->model->enc_conv0_weight, ctx->model->enc_conv0_bias, 1280, 3, 1, c0l));
    x = ggml_gelu_erf(gctx, causal_conv1d_graph(gctx, x, c0l, ctx->model->enc_conv1_weight, ctx->model->enc_conv1_bias, 1280, 3, 2, c1l));
    ggml_tensor * p = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, c1l); ggml_set_name(p, "enc_positions"); ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, p, ctx->backend);
    ggml_tensor * m = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, c1l, c1l); ggml_set_name(m, "enc_attn_mask"); ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, m, ctx->backend);
    x = ggml_reshape_2d(gctx, ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3)), 1280, c1l);
    for (int32_t i = 0; i < 32; i++) {
        auto & L = ctx->model->enc_layers[i];
        ggml_tensor * residual = x, * x_norm = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), L.attn_norm_weight);
        ggml_tensor * q = ggml_add(gctx, ggml_mul_mat(gctx, L.attn_q_weight, x_norm), L.attn_q_bias);
        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm);
        ggml_tensor * v = ggml_add(gctx, ggml_mul_mat(gctx, L.attn_v_weight, x_norm), L.attn_v_bias);
        q = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, q, 64, 32, c1l), p, nullptr, 64, 0, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        k = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, k, 64, 32, c1l), p, nullptr, 64, 0, 0, 1000000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        ggml_tensor * ao = ggml_flash_attn_ext(gctx, ggml_permute(gctx, q, 0, 2, 1, 3), ggml_permute(gctx, k, 0, 2, 1, 3), ggml_permute(gctx, ggml_reshape_3d(gctx, v, 64, 32, c1l), 0, 2, 1, 3), ggml_cast(gctx, m, GGML_TYPE_F16), 1.f/8.f, 0.f, 0.f);
        x = ggml_add(gctx, residual, ggml_add(gctx, ggml_mul_mat(gctx, L.attn_o_weight, ggml_reshape_2d(gctx, ggml_cont(gctx, ao), 2048, c1l)), L.attn_o_bias));
        residual = x; x_norm = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), L.ffn_norm_weight);
        ggml_tensor * ffn = ggml_mul_mat(gctx, L.ffn_w2_weight, ggml_mul(gctx, ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm)), ggml_mul_mat(gctx, L.ffn_w3_weight, x_norm)));
        x = ggml_add(gctx, residual, ggml_add(gctx, ffn, L.ffn_w2_bias));
    }
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), ctx->model->enc_norm_weight);
    if (out_len) *out_len = c1l;
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, ggml_view_2d(gctx, ctx->encoder_chunk_output, 1280, c1l, ctx->encoder_chunk_output->nb[1], 0)));
    return gf;
}

ggml_cgraph * build_adapter_graph_slice(voxtral_context * ctx, ggml_context * gctx, int32_t slot_id, int32_t e_off, int32_t e_count) {
    int32_t dec = e_count / 4; ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * x = ggml_reshape_2d(gctx, ggml_view_2d(gctx, ctx->encoder_output, 1280, e_count, ctx->encoder_output->nb[1], (size_t)slot_id*4000*1280*4 + (size_t)e_off*1280*4), 1280*4, dec);
    x = ggml_mul_mat(gctx, ctx->model->adapter_2_weight, ggml_gelu_erf(gctx, ggml_mul_mat(gctx, ctx->model->adapter_0_weight, x)));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, ggml_view_2d(gctx, ctx->decoder_memory, 3072, dec, ctx->decoder_memory->nb[1], (size_t)slot_id*1000*3072*4 + (size_t)(e_off/4)*3072*4)));
    return gf;
}

ggml_cgraph * build_adapter_graph(voxtral_context * ctx, ggml_context * gctx) {
    int32_t dec = ctx->enc_seq_used / 4; ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * x = ggml_reshape_2d(gctx, ggml_view_2d(gctx, ctx->encoder_output, 1280, ctx->enc_seq_used, ctx->encoder_output->nb[1], 0), 1280*4, dec);
    x = ggml_mul_mat(gctx, ctx->model->adapter_2_weight, ggml_gelu_erf(gctx, ggml_mul_mat(gctx, ctx->model->adapter_0_weight, x)));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, ggml_view_2d(gctx, ctx->decoder_memory, 3072, dec, ctx->decoder_memory->nb[1], 0)));
    return gf;
}
bool run_decoder_step_batched(voxtral_context * ctx, int32_t n, const int32_t * slot_ids, const int32_t * tokens, const int32_t * positions, const int32_t * kv_used, const float ** embs, float * out) {
    const int32_t N = ctx->max_parallel_streams;
    std::vector<int32_t> kvs(N, 0), toks(N, 32), poss(N, 0);
    std::vector<float> embs_flat(N * 3072, 0.f);
    for (int i=0; i<n; i++) {
        int32_t sid = slot_ids[i]; kvs[sid] = kv_used[i]; toks[sid] = tokens[i]; poss[sid] = positions[i];
        if (embs && embs[i]) memcpy(embs_flat.data() + sid * 3072, embs[i], 3072 * 4);
        else {
            ggml_backend_tensor_get(ctx->decoder_memory, embs_flat.data() + sid * 3072, (size_t)sid * 1000 * 3072 * 4 + (size_t)positions[i] * 3072 * 4, 3072 * 4);
        }
    }
    ggml_init_params p = { 4096*1024, nullptr, true }; ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_decoder_step_batched_graph(ctx, gctx, kvs.data());
    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_step, gf)) { ggml_free(gctx); return false; }
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "token_ids"), toks.data(), 0, N * 4);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "positions"), poss.data(), 0, N * 4);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "time_emb"), ctx->time_emb_cpu.data(), 0, 3072 * 4);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "audio_emb"), embs_flat.data(), 0, N * 3072 * 4);
    ggml_tensor * msk = find_tensor_in_graph(gf, "msk");
    if (msk) {
        int32_t mk = (int32_t)msk->ne[0]; std::vector<float> m(mk * N, -1e30f);
        for (int b=0; b<N; b++) { for (int j=0; j<=kvs[b]; j++) m[b*mk + j] = 0.f; }
        ggml_backend_tensor_set(msk, m.data(), 0, m.size() * 4);
    }
    ggml_backend_sched_graph_compute(ctx->sched_dec_step, gf);
    ggml_backend_tensor_get(ctx->decoder_logits, out, 0, N * VOXTRAL_VOCAB_SIZE * 4);
    ggml_free(gctx); return true;
}

bool run_encoder_chunk(voxtral_context * ctx, const float * mel, int32_t frames, int32_t rope_off, int32_t * out_len) {
    ggml_init_params p = { 4096 * 1024, nullptr, true }; ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_encoder_graph(ctx, gctx, mel, frames, out_len);
    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) { ggml_free(gctx); return false; }
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "mel_input"), mel, 0, (size_t)frames * 128 * 4);
    ggml_tensor * pos_t = find_tensor_in_graph(gf, "enc_positions");
    if (pos_t) { std::vector<int32_t> pos(pos_t->ne[0]); for (int i=0; i<(int)pos.size(); i++) pos[i] = rope_off + i; ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * 4); }
    ggml_tensor * msk_t = find_tensor_in_graph(gf, "enc_attn_mask");
    if (msk_t) { int32_t n = (int32_t)msk_t->ne[0]; std::vector<float> m(n * n, 0.f); ggml_backend_tensor_set(msk_t, m.data(), 0, m.size() * 4); }
    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf); ggml_free(gctx); return true;
}

bool run_adapter(voxtral_context * ctx) {
    ggml_init_params p = { 4096 * 1024, nullptr, true }; ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_adapter_graph(ctx, gctx); ggml_backend_sched_reset(ctx->sched_adapter);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_adapter, gf)) { ggml_free(gctx); return false; }
    ggml_backend_sched_graph_compute(ctx->sched_adapter, gf);
    ggml_free(gctx); return true;
}

bool run_decoder_prefill(voxtral_context * ctx, int32_t slot_id, const int32_t * tokens, int32_t n, float * out) {
    ggml_init_params p = { 4096 * 1024, nullptr, true }; ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_decoder_prefill_graph(ctx, gctx, slot_id, n);
    ggml_backend_sched_reset(ctx->sched_dec_pre);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_pre, gf)) { ggml_free(gctx); return false; }
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "token_ids"), tokens, 0, n * 4);
    std::vector<int32_t> pos(n); std::iota(pos.begin(), pos.end(), 0);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "positions"), pos.data(), 0, n * 4);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "time_emb"), ctx->time_emb_cpu.data(), 0, 3072 * 4);
    ggml_tensor * msk = find_tensor_in_graph(gf, "causal_mask");
    if (msk) { std::vector<float> m(n * n); for (int i=0; i<n; i++) for (int j=0; j<n; j++) m[i*n+j] = (j<=i?0.f:-1e30f); ggml_backend_tensor_set(msk, m.data(), 0, m.size() * 4); }
    ggml_backend_sched_graph_compute(ctx->sched_dec_pre, gf);
    ggml_backend_tensor_get(ctx->decoder_logits, out, (size_t)slot_id * VOXTRAL_VOCAB_SIZE * 4, VOXTRAL_VOCAB_SIZE * 4);
    ggml_free(gctx); return true;
}

bool run_decoder_step(voxtral_context * ctx, int32_t slot_id, int32_t token, int32_t pos, int32_t a_pos, int32_t kv_used, float * out) {
    ggml_init_params p = { 4096 * 1024, nullptr, true }; ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_decoder_step_graph(ctx, gctx, slot_id, pos, a_pos);
    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_step, gf)) { ggml_free(gctx); return false; }
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "token_id"), &token, 0, 4);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "position"), &pos, 0, 4);
    ggml_backend_tensor_set(find_tensor_in_graph(gf, "time_emb"), ctx->time_emb_cpu.data(), 0, 3072 * 4);
    ggml_backend_sched_graph_compute(ctx->sched_dec_step, gf);
    ggml_backend_tensor_get(ctx->decoder_logits, out, (size_t)slot_id * VOXTRAL_VOCAB_SIZE * 4, VOXTRAL_VOCAB_SIZE * 4);
    ggml_free(gctx); return true;
}
voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger,
    voxtral_gpu_backend    gpu,
    int32_t                gpu_device)
{
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = { true, &ctx_meta };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) return nullptr;
    voxtral_model * model = new voxtral_model();
    model->gguf_ctx  = gguf_ctx; model->ctx_gguf  = ctx_meta;
    ggml_backend_t weights_backend = nullptr;
    voxtral_gpu_backend resolved_gpu = voxtral_gpu_backend::none;
#ifdef GGML_USE_VULKAN
    if (gpu == voxtral_gpu_backend::vulkan || gpu == voxtral_gpu_backend::auto_detect) {
        weights_backend = ggml_backend_vk_init(gpu_device);
        if (weights_backend) resolved_gpu = voxtral_gpu_backend::vulkan;
    }
#endif
    if (!weights_backend) weights_backend = ggml_backend_cpu_init();
    model->backend_weights = weights_backend;
    model->weights_on_gpu = (resolved_gpu != voxtral_gpu_backend::none);
    model->gpu_type = resolved_gpu;
    model->buf_weights = ggml_backend_alloc_ctx_tensors(ctx_meta, weights_backend);
    {
        FILE * fp = fopen(path.c_str(), "rb");
        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) continue;
            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
            const size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> tmp(nbytes);
            fseek(fp, (long)offset, SEEK_SET);
            fread(tmp.data(), 1, nbytes, fp);
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }
    model->enc_conv0_weight = get_tensor(ctx_meta, "enc.conv0.weight");
    model->enc_conv0_bias   = get_tensor(ctx_meta, "enc.conv0.bias");
    model->enc_conv1_weight = get_tensor(ctx_meta, "enc.conv1.weight");
    model->enc_conv1_bias   = get_tensor(ctx_meta, "enc.conv1.bias");
    model->enc_norm_weight  = get_tensor(ctx_meta, "enc.norm.weight");
    model->enc_layers.resize(32);
    for (int i=0; i<32; i++) {
        char nm[256]; auto & L = model->enc_layers[i];
        snprintf(nm,256,"enc.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_q.bias",i);      L.attn_q_bias = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_v.bias",i);      L.attn_v_bias = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.attn_o.bias",i);      L.attn_o_bias = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.ffn_w2.bias",i);      L.ffn_w2_bias = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"enc.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
    }
    model->adapter_0_weight = get_tensor(ctx_meta, "adapter.0.weight");
    model->adapter_2_weight = get_tensor(ctx_meta, "adapter.2.weight");
    model->tok_embeddings_weight = get_tensor(ctx_meta, "tok_embeddings.weight");
    model->dec_norm_weight = get_tensor(ctx_meta, "norm.weight");
    model->dec_layers.resize(26);
    for (int i=0; i<26; i++) {
        char nm[256]; auto & L = model->dec_layers[i];
        snprintf(nm,256,"dec.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.ada0.weight",i);      L.ada0_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,256,"dec.blk.%d.ada2.weight",i);      L.ada2_weight = get_tensor(ctx_meta,nm);
    }
    model->mel_filters = get_tensor(ctx_meta, "audio.mel_filters");
    {
        const int k_special = gguf_find_key(gguf_ctx, "tokenizer.ggml.special_tokens");
        if (k_special >= 0) {
            const int n = gguf_get_arr_n(gguf_ctx, k_special);
            const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf_ctx, k_special);
            if (data) for (int i=0; i<n; i++) model->tokenizer_special_ranks.insert(data[i]);
        }
        const int k_vocab = gguf_find_key(gguf_ctx, "voxtral.tokenizer.vocab_token_bytes_b64");
        if (k_vocab >= 0) {
            const int n = gguf_get_arr_n(gguf_ctx, k_vocab);
            model->tokenizer_vocab_b64.reserve(n);
            for (int i=0; i<n; i++) model->tokenizer_vocab_b64.emplace_back(gguf_get_arr_str(gguf_ctx, k_vocab, i));
        }
    }
    model->tokenizer_num_special_tokens = 1000;
    return model;
}

void voxtral_model_free(voxtral_model * m) {
    if (!m) return; if (m->buf_weights) ggml_backend_buffer_free(m->buf_weights); if (m->backend_weights) ggml_backend_free(m->backend_weights);
    if (m->ctx_gguf) ggml_free(m->ctx_gguf); if (m->gguf_ctx) gguf_free(m->gguf_ctx); delete m;
}

voxtral_context * voxtral_init_from_model(voxtral_model * model, const voxtral_context_params & params) {
    voxtral_context * ctx = new voxtral_context(); ctx->model = model; ctx->log_level = params.log_level; ctx->logger = params.logger; ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->max_parallel_streams = params.max_parallel_streams > 0 ? params.max_parallel_streams : 1;
    const int32_t N = ctx->max_parallel_streams;

    if (params.gpu == model->gpu_type && model->backend_weights) { ctx->backend = model->backend_weights; ctx->gpu_type = model->gpu_type; }
    else {
#ifdef GGML_USE_VULKAN
        if (params.gpu == voxtral_gpu_backend::vulkan) { ctx->backend = ggml_backend_vk_init(params.gpu_device); if (ctx->backend) ctx->gpu_type = voxtral_gpu_backend::vulkan; }
#endif
    }
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_init_params params_persistent = { 1024 * 1024, nullptr, true };
    ctx->ctx_persistent = ggml_init(params_persistent); if (!ctx->ctx_persistent) return nullptr;
    ctx->kv_self_k = ggml_new_tensor_4d(ctx->ctx_persistent, GGML_TYPE_F32, 1024, 1000, 26, N);
    ctx->kv_self_v = ggml_new_tensor_4d(ctx->ctx_persistent, GGML_TYPE_F32, 1024, 1000, 26, N);
    ctx->encoder_output = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32, 1280, 4000, N);
    ctx->decoder_memory = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32, 3072, 1000, N);
    ctx->decoder_logits = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_VOCAB_SIZE, N);
    ctx->encoder_chunk_output = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32, 1280, 2000);
    ctx->buf_persistent = ggml_backend_alloc_ctx_tensors(ctx->ctx_persistent, ctx->backend);
    if (!ctx->buf_persistent) return nullptr;
    ggml_backend_buffer_clear(ctx->buf_persistent, 0);
    ggml_backend_t bes[] = {ctx->backend, ctx->backend_cpu};
    ctx->sched_encoder = ggml_backend_sched_new(bes, nullptr, 2, 16384, false, true);
    ctx->sched_adapter = ggml_backend_sched_new(bes, nullptr, 2, 16384, false, true);
    ctx->sched_dec_pre = ggml_backend_sched_new(bes, nullptr, 2, 16384, false, true);
    ctx->sched_dec_step = ggml_backend_sched_new(bes, nullptr, 2, 16384, false, true);
    ctx->hann_window.resize(400); for (int i=0; i<400; i++) ctx->hann_window[i] = 0.5f * (1.f - cosf(2.f*VOXTRAL_PI*i/400.f));
    ctx->mel_filters_cpu.resize(201*128); ggml_backend_tensor_get(model->mel_filters, ctx->mel_filters_cpu.data(), 0, 201*128*4);
    compute_time_embedding(ctx->time_emb_cpu, 6.0f, 3072); return ctx;
}

void voxtral_free(voxtral_context * ctx) {
    if (!ctx) return; ggml_backend_sched_free(ctx->sched_encoder); ggml_backend_sched_free(ctx->sched_adapter);
    ggml_backend_sched_free(ctx->sched_dec_pre); ggml_backend_sched_free(ctx->sched_dec_step);
    ggml_backend_buffer_free(ctx->buf_persistent); ggml_free(ctx->ctx_persistent);
    ggml_backend_free(ctx->backend_cpu); ggml_backend_free(ctx->backend); delete ctx;
}

voxtral_stream * voxtral_stream_create(voxtral_context * ctx, int32_t slot_id) { auto * s = new voxtral_stream(); s->ctx = ctx; s->slot_id = slot_id; voxtral_stream_reset(s); return s; }
void voxtral_stream_free(voxtral_stream * s) { delete s; }
void voxtral_stream_reset(voxtral_stream * s) {
    if (!s) return;
    s->audio_buf.assign(48640, 0.f);
    s->samples_processed = 0; s->all_tokens.clear(); s->tokens_reported = 0;
    s->last_token = VOXTRAL_TOKEN_STREAMING_PAD; s->dec_position = 0; s->kv_used = 0; s->enc_tokens_total = 0; s->enc_kv_used = 0;
    s->dec_positions_total = 0; s->consecutive_pad = 0; s->seen_text = false; s->prefilled = false; clear_kv_cache_slotted(s->ctx, s->slot_id);
}

static bool stream_decoder_prefill(voxtral_stream * s) {
    std::vector<int32_t> ids = {VOXTRAL_TOKEN_BOS}; for (int i=0; i<38; i++) ids.push_back(VOXTRAL_TOKEN_STREAMING_PAD);
    std::vector<float> logits(VOXTRAL_VOCAB_SIZE); if (!run_decoder_prefill(s->ctx, s->slot_id, ids.data(), (int)ids.size()-1, logits.data())) return false;
    s->kv_used = (int)ids.size()-1; if (!run_decoder_step(s->ctx, s->slot_id, ids.back(), (int)ids.size()-1, (int)ids.size()-1, s->kv_used, logits.data())) return false;
    s->kv_used++; int32_t t = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    s->all_tokens.push_back(t); s->last_token = t; s->dec_position = (int)ids.size(); s->prefilled = true; return true;
}

bool stream_process_audio_to_encoder(voxtral_stream * s, const float * a, int32_t n) {
    if (n > 0) s->audio_buf.insert(s->audio_buf.end(), a, a + n);
    while (true) {
        int32_t ms = s->samples_processed;
        int32_t avail = (int)s->audio_buf.size() - ms;
        if (avail < 51200) break; // 3.2s
        
        int32_t nf = 3000; std::vector<float> mel(128 * nf);
        compute_mel_spectrogram(s->audio_buf.data() + ms, 51200, s->ctx->mel_filters_cpu.data(), s->ctx->hann_window.data(), mel.data(), &nf);
        
        int32_t e_len = 0; if (!run_encoder_chunk(s->ctx, mel.data(), nf, 0, &e_len)) return false;
        
        int32_t e_count = 160; 
        std::vector<uint8_t> tmp(e_count * 1280 * 4); ggml_backend_tensor_get(s->ctx->encoder_chunk_output, tmp.data(), 0, tmp.size());
        int32_t abs_enc = (ms / 320);
        ggml_backend_tensor_set(s->ctx->encoder_output, tmp.data(), (size_t)s->slot_id * 4000 * 1280 * 4 + (size_t)abs_enc * 1280 * 4, tmp.size());
        
        ggml_context * gctx_ada = ggml_init({1024*1024, nullptr, true});
        ggml_cgraph * gf_ada = build_adapter_graph_slice(s->ctx, gctx_ada, s->slot_id, abs_enc, e_count);
        ggml_backend_sched_reset(s->ctx->sched_adapter); if (!ggml_backend_sched_alloc_graph(s->ctx->sched_adapter, gf_ada)) { ggml_free(gctx_ada); return false; }
        ggml_backend_sched_graph_compute(s->ctx->sched_adapter, gf_ada); ggml_free(gctx_ada);
        
        s->samples_processed += 51200; s->enc_tokens_total = (s->samples_processed / 320); s->dec_positions_total = s->enc_tokens_total / 4;
    }
    if (!s->prefilled && s->dec_positions_total >= 39) stream_decoder_prefill(s); return s->prefilled;
}

bool stream_decode_available(voxtral_stream * s, std::string & text, bool early) {
    while (s->dec_position < s->dec_positions_total) {
        std::vector<float> lgt(VOXTRAL_VOCAB_SIZE); if (!run_decoder_step(s->ctx, s->slot_id, s->last_token, s->dec_position, s->dec_position, s->kv_used, lgt.data())) return false;
        if (s->kv_used < 1000) s->kv_used++; int32_t t = (int32_t)(std::max_element(lgt.begin(), lgt.end()) - lgt.begin());
        s->all_tokens.push_back(t); s->last_token = t; s->dec_position++; if (t == VOXTRAL_TOKEN_EOS) { s->dec_position = s->dec_positions_total; break; }
    }
    if ((int)s->all_tokens.size() > s->tokens_reported) {
        std::vector<int32_t> nt(s->all_tokens.begin() + s->tokens_reported, s->all_tokens.end());
        text = decode_tokens(*s->ctx->model, nt); s->tokens_reported = (int)s->all_tokens.size();
    }
    return true;
}

bool voxtral_stream_feed(voxtral_stream * s, const float * a, int32_t n, std::string & text) { text.clear(); if (!stream_process_audio_to_encoder(s, a, n)) return true; return stream_decode_available(s, text, false); }
bool voxtral_stream_feed_batched(voxtral_context * ctx, voxtral_stream ** ss, const float ** as, int32_t * ns, int32_t n_ss, std::string * ts) {
    const int32_t N = ctx->max_parallel_streams;
    for (int i = 0; i < n_ss; i++) { ts[i].clear(); stream_process_audio_to_encoder(ss[i], as[i], ns[i]); }
    int32_t steps = 0;
    while (steps < 32) {
        std::vector<int32_t> act; for (int i = 0; i < n_ss; i++) { if (ss[i]->prefilled && ss[i]->dec_position < ss[i]->dec_positions_total) act.push_back(i); }
        if (act.empty()) break;
        std::vector<int32_t> sids(N); for (int i=0; i<N; i++) sids[i]=i;
        std::vector<int32_t> toks(N, 32), poss(N, 0), kvs(N, 0);
        for (int i : act) { int32_t sid = ss[i]->slot_id; toks[sid]=ss[i]->last_token; poss[sid]=ss[i]->dec_position; kvs[sid]=ss[i]->kv_used; }
        std::vector<float> lbs(VOXTRAL_VOCAB_SIZE * N); if (!run_decoder_step_batched(ctx, (int)act.size(), sids.data(), toks.data(), poss.data(), kvs.data(), nullptr, lbs.data())) return false;
        for (int i : act) {
            voxtral_stream * s = ss[i]; int32_t sid = s->slot_id; if (s->kv_used < 1000) s->kv_used++;
            const float * l = lbs.data() + (size_t)sid * VOXTRAL_VOCAB_SIZE;
            int32_t t = (int32_t)(std::max_element(l, l + VOXTRAL_VOCAB_SIZE) - l);
            s->all_tokens.push_back(t); s->last_token = t; s->dec_position++; if (t == VOXTRAL_TOKEN_EOS) s->dec_position = s->dec_positions_total;
        }
        steps++;
    }
    for (int i = 0; i < n_ss; i++) {
        if ((int)ss[i]->all_tokens.size() > ss[i]->tokens_reported) {
            std::vector<int32_t> nt(ss[i]->all_tokens.begin() + ss[i]->tokens_reported, ss[i]->all_tokens.end());
            ts[i] = decode_tokens(*ctx->model, nt); ss[i]->tokens_reported = (int)ss[i]->all_tokens.size();
        }
    }
    return true;
}

bool voxtral_stream_flush(voxtral_stream * s, std::string & text) { text.clear(); return true; }
bool voxtral_stream_warmup(voxtral_stream * s) { return true; }
bool voxtral_debug_copy_encoder_output(const voxtral_context * ctx, int32_t slot_id, std::vector<float> & out) { out.resize(1280 * 4000); ggml_backend_tensor_get(ctx->encoder_output, out.data(), (size_t)slot_id * ctx->encoder_output->nb[2], out.size() * 4); return true; }
bool voxtral_transcribe_audio(voxtral_context & ctx, const std::vector<float> & audio, int32_t max_tokens, voxtral_result & result) { return false; }
bool voxtral_transcribe_file(voxtral_context & ctx, const std::string & path, int32_t max_tokens, voxtral_result & result) { return false; }
bool run_encoder_chunk_kv(voxtral_context * c, const float * m, int32_t f, int32_t k, int32_t * l) { return false; }
