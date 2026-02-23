#pragma once

#include <vksdl/error.hpp>

#include <cassert>
#include <optional>
#include <utility>
#include <variant>

namespace vksdl {

// Result<T> holds either a value or an Error.
// Intentionally simple -- no monadic chaining, just check and unwrap.
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}             // NOLINT implicit
    Result(Error error) : data_(std::move(error)) {}         // NOLINT implicit

    [[nodiscard]] bool ok() const { return std::holds_alternative<T>(data_); }
    explicit operator bool() const { return ok(); }

    [[nodiscard]] T& value() & {
        assert(ok() && "called value() on error Result");
        return std::get<T>(data_);
    }

    [[nodiscard]] const T& value() const& {
        assert(ok() && "called value() on error Result");
        return std::get<T>(data_);
    }

    [[nodiscard]] T value() && {
        assert(ok() && "called value() on error Result");
        return std::move(std::get<T>(data_));
    }

    [[nodiscard]] Error& error() & {
        assert(!ok() && "called error() on ok Result");
        return std::get<Error>(data_);
    }

    [[nodiscard]] const Error& error() const& {
        assert(!ok() && "called error() on ok Result");
        return std::get<Error>(data_);
    }

    [[nodiscard]] Error error() && {
        assert(!ok() && "called error() on ok Result");
        return std::move(std::get<Error>(data_));
    }

    // Unwrap or throw. Says what it does.
    [[nodiscard]] T orThrow() && {
        if (!ok()) throwError(error());
        return std::move(std::get<T>(data_));
    }

private:
    std::variant<T, Error> data_;
};

// Specialization for Result<void> -- success carries no value.
template <>
class Result<void> {
public:
    Result() : error_{} {}                                      // success
    Result(Error error) : error_(std::move(error)) {}           // NOLINT implicit

    [[nodiscard]] bool ok() const { return !error_.has_value(); }
    explicit operator bool() const { return ok(); }

    [[nodiscard]] Error& error() & {
        assert(!ok() && "called error() on ok Result<void>");
        return *error_;
    }

    [[nodiscard]] const Error& error() const& {
        assert(!ok() && "called error() on ok Result<void>");
        return *error_;
    }

    void orThrow() && {
        if (!ok()) throwError(error());
    }

private:
    std::optional<Error> error_;
};

} // namespace vksdl
