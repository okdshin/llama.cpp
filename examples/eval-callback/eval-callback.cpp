#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

#include <fstream>
#include <sstream>
#include <mutex>

static std::mutex file_mutex;

void writeUint64LE(std::ofstream& file, uint64_t value) {
    uint8_t bytes[8];
    bytes[0] = (value >> 0) & 0xFF;
    bytes[1] = (value >> 8) & 0xFF;
    bytes[2] = (value >> 16) & 0xFF;
    bytes[3] = (value >> 24) & 0xFF;
    bytes[4] = (value >> 32) & 0xFF;
    bytes[5] = (value >> 40) & 0xFF;
    bytes[6] = (value >> 48) & 0xFF;
    bytes[7] = (value >> 56) & 0xFF;
    file.write(reinterpret_cast<const char*>(bytes), 8);
}

class JsonBuilder {
    std::ostringstream ss;
    bool first_field = true;
public:
    JsonBuilder() { ss << "{"; }

    void addField(const std::string& key, const std::string& value) {
        if (!first_field) ss << ",";
        ss << "\"" << key << "\":\"" << value << "\"";
        first_field = false;
    }

    void addArray(const std::string& key, const std::vector<size_t>& arr) {
        if (!first_field) ss << ",";
        ss << "\"" << key << "\":[";
        for (size_t i = 0; i < arr.size(); ++i) {
            ss << arr[i];
            if (i < arr.size() - 1) ss << ",";
        }
        ss << "]";
        first_field = false;
    }

    void addObject(const std::string& key, const std::string& json_obj) {
        if (!first_field) ss << ",";
        ss << "\"" << key << "\":" << json_obj;
        first_field = false;
    }

    void addEmptyObject(const std::string& key) {
        if (!first_field) ss << ",";
        ss << "\"" << key << "\":{}";
        first_field = false;
    }

    std::string str() const {
        return ss.str() + "}";
    }
};

template<typename T>
bool saveSafetensor(const std::string& filename,
                   const std::string& tensor_name,
                   const std::vector<size_t>& shape,
                   const std::string& dtype,
                   const T* data) {
    // Lock for file operations
    std::lock_guard<std::mutex> lock(file_mutex);
    try {
        // Calculate sizes before acquiring lock
        size_t elem_size;
        if (dtype == "F32") elem_size = 4;
        else if (dtype == "F16") elem_size = 2;
        else if (dtype == "I32") elem_size = 4;
        else if (dtype == "I8") elem_size = 1;
        else throw std::runtime_error("Unsupported dtype: " + dtype);

        size_t total_elements = 1;
        for (size_t dim : shape) {
            total_elements *= dim;
        }
        size_t tensor_size = total_elements * elem_size;

        // Prepare JSON header before acquiring lock
        JsonBuilder tensor_info;
        tensor_info.addField("dtype", dtype);
        tensor_info.addArray("shape", shape);
        tensor_info.addArray("data_offsets", {0, tensor_size});

        JsonBuilder root;
        root.addObject(tensor_name, tensor_info.str());
        root.addEmptyObject("__metadata__");

        std::string header = root.str();
        uint64_t header_size = header.size();


        // Open file with truncation mode to ensure clean write
        std::ofstream file(filename, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!file) {
            return false;
        }

        // Ensure the file is empty
        file.seekp(0);

        // Write header size in little endian
        writeUint64LE(file, header_size);

        // Write header JSON
        file.write(header.data(), header.size());

        // Write tensor data
        file.write(reinterpret_cast<const char*>(data), tensor_size);

        // Ensure all data is written
        file.flush();
        file.close();

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0->name, ggml_ne_string(src0).c_str(),
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);

        std::vector<size_t> shape;
        if(ggml_n_dims(t) == 4) {
            shape = {static_cast<size_t>(t->ne[3]), static_cast<size_t>(t->ne[2]), static_cast<size_t>(t->ne[1]), static_cast<size_t>(t->ne[0])};
        }
        else if(ggml_n_dims(t) == 3) {
            shape = {static_cast<size_t>(t->ne[2]), static_cast<size_t>(t->ne[1]), static_cast<size_t>(t->ne[0])};
        }
        else if(ggml_n_dims(t) == 2) {
            shape = {static_cast<size_t>(t->ne[1]), static_cast<size_t>(t->ne[0])};
        }
        else if(ggml_n_dims(t) == 1) {
            shape = {static_cast<size_t>(t->ne[0])};
        }

        saveSafetensor(
            std::string(t->name) + ".safetensors",
            t->name,
            shape,
            "F32",
            static_cast<float*>(t->data)
        );
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    for(auto t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "\n";
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
