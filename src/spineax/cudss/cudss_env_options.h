#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "cudss.h"

static int cudss_ir_nsteps() {
    const char* raw = std::getenv("SPINEAX_CUDSS_IR_N_STEPS");
    if (raw == nullptr || raw[0] == '\0') return 5;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) return 5;
    return static_cast<int>(value);
}

static std::string cudss_normalize_env_value(const char* raw) {
    std::string value(raw == nullptr ? "" : raw);
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

struct CudssEnvAlgOption {
    bool should_set = false;
    cudssAlgType_t value = CUDSS_ALG_DEFAULT;
    std::string error;
};

struct CudssEnvBoolOption {
    bool should_set = false;
    int value = 0;
    std::string error;
};

static CudssEnvAlgOption cudss_parse_alg_env(const char* name) {
    CudssEnvAlgOption result;
    const char* raw = std::getenv(name);
    if (raw == nullptr) return result;
    std::string value = cudss_normalize_env_value(raw);
    if (value.empty() || value == "default" || value == "alg_default") return result;
    if (value == "0") {
        result.should_set = true;
        result.value = CUDSS_ALG_DEFAULT;
        return result;
    }
    if (value.rfind("alg_", 0) == 0) {
        value = value.substr(4);
    }
    char* end = nullptr;
    long alg = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        result.error = std::string("Invalid ") + name + " value '" + raw +
                       "'. Accepted values: unset, empty, default, 0, 1, 2, 3, 4, 5, alg_1, alg_2, alg_3, alg_4, alg_5.";
        return result;
    }
    switch (alg) {
        case 1: result.value = CUDSS_ALG_1; break;
        case 2: result.value = CUDSS_ALG_2; break;
        case 3: result.value = CUDSS_ALG_3; break;
        case 4: result.value = CUDSS_ALG_4; break;
        case 5: result.value = CUDSS_ALG_5; break;
        default:
            result.error = std::string("Invalid ") + name + " value '" + raw +
                           "'. Accepted values: unset, empty, default, 0, 1, 2, 3, 4, 5, alg_1, alg_2, alg_3, alg_4, alg_5.";
            return result;
    }
    result.should_set = true;
    return result;
}

static CudssEnvBoolOption cudss_parse_bool_env(const char* name) {
    CudssEnvBoolOption result;
    const char* raw = std::getenv(name);
    if (raw == nullptr) return result;
    std::string value = cudss_normalize_env_value(raw);
    if (value.empty() || value == "default") return result;
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        result.should_set = true;
        result.value = 1;
        return result;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        result.should_set = true;
        result.value = 0;
        return result;
    }
    result.error = std::string("Invalid ") + name + " value '" + raw +
                   "'. Accepted values: unset, empty, default, 0, 1, false, true, no, yes, off, on.";
    return result;
}

static xla::ffi::Error cudss_apply_env_options_or_error(cudssConfig_t config) {
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    CudssEnvAlgOption alg = cudss_parse_alg_env("SPINEAX_CUDSS_REORDERING_ALG");
    if (!alg.error.empty()) return xla::ffi::Error::InvalidArgument(alg.error);
    if (alg.should_set) {
        status = cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg.value, sizeof(alg.value));
        if (status != CUDSS_STATUS_SUCCESS) {
            return xla::ffi::Error::Internal(
                std::string("cuDSS call failed with status ") + std::to_string(status) +
                ": cudssConfigSet SPINEAX_CUDSS_REORDERING_ALG");
        }
    }

    alg = cudss_parse_alg_env("SPINEAX_CUDSS_FACTORIZATION_ALG");
    if (!alg.error.empty()) return xla::ffi::Error::InvalidArgument(alg.error);
    if (alg.should_set) {
        status = cudssConfigSet(config, CUDSS_CONFIG_FACTORIZATION_ALG, &alg.value, sizeof(alg.value));
        if (status != CUDSS_STATUS_SUCCESS) {
            return xla::ffi::Error::Internal(
                std::string("cuDSS call failed with status ") + std::to_string(status) +
                ": cudssConfigSet SPINEAX_CUDSS_FACTORIZATION_ALG");
        }
    }

    CudssEnvBoolOption deterministic = cudss_parse_bool_env("SPINEAX_CUDSS_DETERMINISTIC_MODE");
    if (!deterministic.error.empty()) return xla::ffi::Error::InvalidArgument(deterministic.error);
    if (deterministic.should_set) {
        status = cudssConfigSet(config, CUDSS_CONFIG_DETERMINISTIC_MODE,
                                &deterministic.value, sizeof(deterministic.value));
        if (status != CUDSS_STATUS_SUCCESS) {
            return xla::ffi::Error::Internal(
                std::string("cuDSS call failed with status ") + std::to_string(status) +
                ": cudssConfigSet SPINEAX_CUDSS_DETERMINISTIC_MODE");
        }
    }
    return xla::ffi::Error::Success();
}
