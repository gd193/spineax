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

static bool cudss_parse_alg_env(const char* name, cudssAlgType_t* out) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return false;
    std::string value = cudss_normalize_env_value(raw);
    if (value == "default" || value == "alg_default" || value == "0") {
        *out = CUDSS_ALG_DEFAULT;
        return true;
    }
    if (value.rfind("alg_", 0) == 0) {
        value = value.substr(4);
    }
    char* end = nullptr;
    long alg = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') return false;
    switch (alg) {
        case 1: *out = CUDSS_ALG_1; return true;
        case 2: *out = CUDSS_ALG_2; return true;
        case 3: *out = CUDSS_ALG_3; return true;
        case 4: *out = CUDSS_ALG_4; return true;
        case 5: *out = CUDSS_ALG_5; return true;
        default: return false;
    }
}

static bool cudss_parse_bool_env(const char* name, int* out) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return false;
    std::string value = cudss_normalize_env_value(raw);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        *out = 1;
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        *out = 0;
        return true;
    }
    return false;
}

static cudssStatus_t cudss_apply_env_options(cudssConfig_t config) {
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    cudssAlgType_t alg;
    if (cudss_parse_alg_env("SPINEAX_CUDSS_REORDERING_ALG", &alg)) {
        status = cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg));
        if (status != CUDSS_STATUS_SUCCESS) return status;
    }
    if (cudss_parse_alg_env("SPINEAX_CUDSS_FACTORIZATION_ALG", &alg)) {
        status = cudssConfigSet(config, CUDSS_CONFIG_FACTORIZATION_ALG, &alg, sizeof(alg));
        if (status != CUDSS_STATUS_SUCCESS) return status;
    }
    int deterministic = 0;
    if (cudss_parse_bool_env("SPINEAX_CUDSS_DETERMINISTIC_MODE", &deterministic)) {
        status = cudssConfigSet(config, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic, sizeof(deterministic));
        if (status != CUDSS_STATUS_SUCCESS) return status;
    }
    return status;
}
