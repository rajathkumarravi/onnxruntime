// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <memory>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"

#include "gtest/gtest.h"

class ExperimentalCApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ASSERT_NE(api_, nullptr);
  }

  const OrtApi* api_ = nullptr;
};

// Null name returns nullptr
TEST_F(ExperimentalCApiTest, NullNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction(nullptr);
  EXPECT_EQ(fn, nullptr);
}

// Unknown name returns nullptr
TEST_F(ExperimentalCApiTest, UnknownNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction("NonExistentFunction");
  EXPECT_EQ(fn, nullptr);
}

// Empty string returns nullptr
TEST_F(ExperimentalCApiTest, EmptyNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction("");
  EXPECT_EQ(fn, nullptr);
}

// Known name resolves to non-null (C-style lookup)
TEST_F(ExperimentalCApiTest, KnownNameResolvesC) {
  OrtExperimentalFnPtr fn =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  EXPECT_NE(fn, nullptr);
}

// Known name resolves to non-null (C++-style accessor).
// The typed accessor's nullptr path (unknown name) is covered transitively via the C-side
// UnknownNameReturnsNull test, which exercises the same underlying GetExperimentalFunction lookup.
TEST_F(ExperimentalCApiTest, KnownNameResolvesCpp) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_);
  EXPECT_NE(fn, nullptr);
}

// Call through typed pointer succeeds and returns the expected sentinel value.
// Uses the throwing accessor (FnOrThrow), which returns a guaranteed-non-null pointer or throws.
// The throw-on-unavailable path is not directly unit-testable: every .inc entry is registered, so a valid
// typed accessor never observes a nullptr. Availability checking is covered by the nullable-accessor tests above.
TEST_F(ExperimentalCApiTest, CallThroughTypedPointer) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_FnOrThrow(api_);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn, Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_));

  int64_t result = 0;
  auto status = Ort::Status{fn(&result)};
  ASSERT_TRUE(status.IsOK()) << "ExperimentalApiTest returned failure status: " << status.GetErrorMessage();
  EXPECT_EQ(result, 12345);
}

// The same name looked up twice returns the same pointer
TEST_F(ExperimentalCApiTest, ConsistentLookup) {
  OrtExperimentalFnPtr fn1 =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  OrtExperimentalFnPtr fn2 =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  EXPECT_EQ(fn1, fn2);
}

#if !defined(ORT_MINIMAL_BUILD)
TEST_F(ExperimentalCApiTest, PromotedEpContextAliasesResolveToStableFunctions) {
  const auto* ep_api = api_->GetEpApi();
  const auto* compile_api = api_->GetCompileApi();
  ASSERT_NE(ep_api, nullptr);
  ASSERT_NE(compile_api, nullptr);

  struct Alias {
    const char* name;
    OrtExperimentalFnPtr stable_function;
  };
  const std::array aliases{
      Alias{"OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28",
            reinterpret_cast<OrtExperimentalFnPtr>(api_->SessionOptionsSetEpContextDataReadFunc)},
      Alias{"OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28",
            reinterpret_cast<OrtExperimentalFnPtr>(compile_api->ModelCompilationOptions_SetEpContextDataWriteFunc)},
      Alias{"OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28",
            reinterpret_cast<OrtExperimentalFnPtr>(ep_api->SessionOptionsGetEpContextConfig)},
      Alias{"OrtEpApi_ReleaseEpContextConfig_SinceV28",
            reinterpret_cast<OrtExperimentalFnPtr>(ep_api->ReleaseEpContextConfig)},
      Alias{"OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28",
            reinterpret_cast<OrtExperimentalFnPtr>(ep_api->EpContextConfigGetEpContextDataReadFunc)},
      Alias{"OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28",
            reinterpret_cast<OrtExperimentalFnPtr>(ep_api->EpContextConfigGetEpContextDataWriteFunc)},
  };
  for (const auto& alias : aliases) {
    SCOPED_TRACE(alias.name);
    EXPECT_EQ(api_->GetExperimentalFunction(alias.name), alias.stable_function);
  }
  EXPECT_EQ(api_->GetExperimentalFunction("OrtEpApi_ReleaseEpContextConfig_SinceV29"), nullptr);
  EXPECT_EQ(api_->GetExperimentalFunction("OrtEpApi_ReleaseEpContextConfig"), nullptr);
}

TEST_F(ExperimentalCApiTest, PromotedEpContextAliasesPreserveConfigLifetimeAndErrors) {
  const auto* ep_api = api_->GetEpApi();
  ASSERT_NE(ep_api, nullptr);
  auto create_config = reinterpret_cast<decltype(ep_api->SessionOptionsGetEpContextConfig)>(
      api_->GetExperimentalFunction("OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28"));
  auto release_config = reinterpret_cast<decltype(ep_api->ReleaseEpContextConfig)>(
      api_->GetExperimentalFunction("OrtEpApi_ReleaseEpContextConfig_SinceV28"));
  auto get_read = reinterpret_cast<decltype(ep_api->EpContextConfigGetEpContextDataReadFunc)>(
      api_->GetExperimentalFunction("OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28"));
  auto get_write = reinterpret_cast<decltype(ep_api->EpContextConfigGetEpContextDataWriteFunc)>(
      api_->GetExperimentalFunction("OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28"));
  ASSERT_NE(create_config, nullptr);
  ASSERT_NE(release_config, nullptr);
  ASSERT_NE(get_read, nullptr);
  ASSERT_NE(get_write, nullptr);

  Ort::SessionOptions options;
  OrtEpContextConfig* raw_config = nullptr;
  Ort::ThrowOnError(create_config(options, &raw_config));
  std::unique_ptr<OrtEpContextConfig, decltype(release_config)> config{raw_config, release_config};
  ASSERT_NE(config, nullptr);

  OrtReadNamedBufferFunc read_func = nullptr;
  OrtWriteNamedBufferFunc write_func = nullptr;
  void* state = &options;
  Ort::ThrowOnError(get_read(config.get(), &read_func, &state));
  EXPECT_EQ(read_func, nullptr);
  EXPECT_EQ(state, nullptr);
  state = &options;
  Ort::ThrowOnError(get_write(config.get(), &write_func, &state));
  EXPECT_EQ(write_func, nullptr);
  EXPECT_EQ(state, nullptr);

  Ort::Status status{get_read(nullptr, &read_func, &state)};
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  release_config(nullptr);
}
#endif
