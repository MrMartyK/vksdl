#include <vksdl/result.hpp>

#include <cassert>
#ifndef VKSDL_ENABLE_EXCEPTIONS
#define VKSDL_ENABLE_EXCEPTIONS 1
#endif
#if VKSDL_ENABLE_EXCEPTIONS
#include <stdexcept>
#endif
#include <string>

int main() {
    // Ok result
    {
        vksdl::Result<int> r = 42;
        assert(r.ok());
        assert(r);
        assert(r.value() == 42);
    }

    // Error result
    {
        vksdl::Result<int> r = vksdl::Error{"create instance", -1, "out of memory"};
        assert(!r.ok());
        assert(!r);
        assert(r.error().operation == "create instance");
        assert(r.error().vkResult == -1);
    }

    // String value
    {
        vksdl::Result<std::string> r = std::string("hello");
        assert(r.ok());
        assert(r.value() == "hello");
    }

    // Returned from function
    {
        auto make = [](bool succeed) -> vksdl::Result<int> {
            if (succeed)
                return 7;
            return vksdl::Error{"test op", 0, "nope"};
        };

        auto good = make(true);
        auto bad = make(false);
        assert(good.ok() && good.value() == 7);
        assert(!bad.ok() && bad.error().message == "nope");
    }

    // Result<void> success
    {
        vksdl::Result<void> r;
        assert(r.ok());
        assert(r);
    }

    // Result<void> error
    {
        vksdl::Result<void> r = vksdl::Error{"compile", -3, "failed"};
        assert(!r.ok());
        assert(!r);
        assert(r.error().operation == "compile");
        assert(r.error().vkResult == -3);
        assert(r.error().message == "failed");
    }

    // Result<void> returned from function
    {
        auto attempt = [](bool succeed) -> vksdl::Result<void> {
            if (succeed)
                return {};
            return vksdl::Error{"op", 0, "nope"};
        };

        assert(attempt(true).ok());
        assert(!attempt(false).ok());
    }

    // orThrow on success
    {
        auto make = []() -> vksdl::Result<int> { return 99; };
        int val = make().orThrow();
        assert(val == 99);
    }

#if VKSDL_ENABLE_EXCEPTIONS
    // orThrow on error (exceptions-enabled builds only).
    {
        auto make = []() -> vksdl::Result<int> { return vksdl::Error{"test", -1, "boom"}; };
        bool caught = false;
        try {
            make().orThrow();
        } catch (const std::runtime_error& e) {
            caught = true;
            std::string msg = e.what();
            assert(msg.find("test") != std::string::npos);
            assert(msg.find("boom") != std::string::npos);
        }
        assert(caught);
    }
#endif

    // orThrow on Result<void> success
    {
        vksdl::Result<void> r;
        std::move(r).orThrow();
    }

#if VKSDL_ENABLE_EXCEPTIONS
    // orThrow on Result<void> error (exceptions-enabled builds only).
    {
        vksdl::Result<void> r = vksdl::Error{"compile", -3, "failed"};
        bool caught = false;
        try {
            std::move(r).orThrow();
        } catch (const std::runtime_error& e) {
            caught = true;
            std::string msg = e.what();
            assert(msg.find("compile") != std::string::npos);
        }
        assert(caught);
    }
#endif

    return 0;
}
