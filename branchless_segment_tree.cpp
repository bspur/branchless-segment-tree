#include <immintrin.h>

#pragma GCC optimize("Ofast,unroll-loops,inline")


#ifdef __clang__
    // #pragma GCC target() doesn't work for Clang for templates
    // minimal == "bmi,bmi2,lzcnt,popcnt"
    // znver3 manual substitute due to Clang bug
    // (Clang doesn't properly support target architectures)
    // NOTE: you can wrap push/pop specifically around your definition
    #pragma clang attribute push (__attribute__((target("avx2,avx,bmi,bmi2,lzcnt,popcnt,sse4.2,sse4.1,ssse3,sse3,sse2,sse"))), apply_to = function)
#else
    #pragma GCC push_options
    #pragma GCC target("arch=znver3,bmi,bmi2,lzcnt,popcnt")
#endif

using namespace std;

using ll = long long;
using uint = unsigned int;

constexpr size_t CACHE_LINE_SIZE = hardware_constructive_interference_size;

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = true_type;
    static constexpr size_t alignment = max(CACHE_LINE_SIZE, alignof(T));
    template <typename U> struct rebind { using other = AlignedAllocator<U>; };
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T), align_val_t(alignment)));
    }
    void deallocate(T* p, size_t n) noexcept {
        ::operator delete(p, n * sizeof(T), align_val_t(alignment));
    }
    // intercept 0-argument construction with default initialization
    // to prevent zero-filling
    template <typename U>
    void construct(U* p) {
        ::new(static_cast<void*>(p)) U; 
    }
};

template <typename E, typename T>
concept ValidIdentity = 
    is_null_pointer_v<E>     ||         // Case 1: defaults to T{}
    is_invocable_r_v<T, E>   ||         // Case 2: functor/lambda/function
    is_convertible_v<E, T>;             // Case 3: structural type

template <typename C, typename T>
concept ValidCombine =
    is_null_pointer_v<C>                           || // plus{}
    is_invocable_r_v<T, C, T, T>                   || // combine(T, T)
    is_invocable_r_v<T, C, T, T, uint>             || // combine(T, T, idx)
    is_invocable_r_v<T, C, T, T, uint, uint>       || // combine(T, T, lsz, rsz)
    is_invocable_r_v<T, C, T, T, uint, uint, uint>;   // combine(T, T, idx, lsz, rsz)

// STL-style transparent functors
#define DEFINE_TRANSPARENT_OP(StructName, HomogeneousExpr, ForwardingExpr) \
template <typename T = void> \
struct StructName { \
    constexpr T operator()(const T& a, const T& b) const { return (HomogeneousExpr); } \
}; \
template <> \
struct StructName<void> { \
    using is_transparent = void; \
    template <typename T, typename U> \
    constexpr auto operator()(T&& a, U&& b) const \
        noexcept(noexcept(ForwardingExpr)) -> decltype(ForwardingExpr) \
    { return (ForwardingExpr); } \
}
DEFINE_TRANSPARENT_OP(assignment, b, std::forward<U>(b));
DEFINE_TRANSPARENT_OP(maximum, a < b ? b : a, a < b ? std::forward<U>(b) : std::forward<T>(a));
DEFINE_TRANSPARENT_OP(minimum, a > b ? b : a, a > b ? std::forward<U>(b) : std::forward<T>(a));
#undef DEFINE_TRANSPARENT_OP

struct missing_predicate {
    using is_transparent = void;
    template <typename T>
    constexpr bool operator()(T&&) const {
        static_assert(sizeof(T) < 0, "Missing search predicate!");
        return false;
    }
};

template <typename T>
concept DecayedType = is_same_v<T, decay_t<T>>;

// flat backpack architecture for markers
struct NullInv {};

template <
    DecayedType Op, DecayedType Inv = NullInv, bool Comm = false,
    bool Single = false, bool EarlyExit = false, bool Heavy = false
>
struct MarkedOp {
    using base_op = Op;
    static constexpr bool is_comm = Comm;
    static constexpr bool is_single = Single;
    static constexpr bool is_early_exit = EarlyExit;
    static constexpr bool is_inv = !is_same_v<Inv, NullInv>;
    static constexpr bool is_heavy = Heavy;

    [[no_unique_address]] Op op;
    [[no_unique_address]] Inv inv_op;

    template<typename... Args> [[gnu::always_inline]] 
    constexpr auto operator()(Args&&... args) const
        noexcept(noexcept(op(std::forward<Args>(args)...)))
        -> decltype(op(std::forward<Args>(args)...))
    { return op(std::forward<Args>(args)...); }

    template<typename... Args> [[gnu::always_inline]] 
    constexpr auto operator()(Args&&... args)
        noexcept(noexcept(op(std::forward<Args>(args)...)))
        -> decltype(op(std::forward<Args>(args)...))
    { return op(std::forward<Args>(args)...); }
};

template <typename T>
constexpr auto commutative(T&& f) {
    using D = decay_t<T>;
    if constexpr (requires { D::is_comm; }) {
        return MarkedOp<
            typename D::base_op, decltype(f.inv_op), true, D::is_single, D::is_early_exit, D::is_heavy
        > { std::forward<T>(f).op, std::forward<T>(f).inv_op };
    } else {
        return MarkedOp<D, NullInv, true, false, false, false>{std::forward<T>(f), {}};
    }
}
template <typename T>
constexpr auto single(T&& f) {
    using D = decay_t<T>;
    if constexpr (requires { D::is_single; }) {
        return MarkedOp<
            typename D::base_op, decltype(f.inv_op), D::is_comm, true, D::is_early_exit, D::is_heavy
        >{ std::forward<T>(f).op, std::forward<T>(f).inv_op };
    } else {
        return MarkedOp<D, NullInv, false, true, false, false>{std::forward<T>(f), {}};
    }
}
template <typename T, typename Inv>
constexpr auto invertible(T&& f, Inv&& inv) {
    using D = std::decay_t<T>;
    if constexpr (requires { D::is_inv; }) {
        return MarkedOp<
            typename D::base_op, decay_t<Inv>, D::is_comm, D::is_single, D::is_early_exit, D::is_heavy
        >{ std::forward<T>(f).op, std::forward<Inv>(inv) };
    } else {
        return MarkedOp<D, decay_t<Inv>, false, false, false, false>{
            std::forward<T>(f), std::forward<Inv>(inv)
        };
    }
}
template <typename T>
constexpr auto early_exit(T&& f) {
    using D = decay_t<T>;
    if constexpr (requires { D::is_early_exit; }) {
        return MarkedOp<
            typename D::base_op, decltype(f.inv_op), D::is_comm, D::is_single, true, D::is_heavy
        >{ std::forward<T>(f).op, std::forward<T>(f).inv_op };
    } else {
        return MarkedOp<D, NullInv, false, false, true, false>{std::forward<T>(f), {}};
    }
}
template <typename T>
constexpr auto heavyweight(T&& f) {
    using D = decay_t<T>;
    if constexpr (requires { D::is_heavy; }) {
        return MarkedOp<
            typename D::base_op, decltype(f.inv_op), D::is_comm, D::is_single, D::is_early_exit, true
        >{ std::forward<T>(f).op, std::forward<T>(f).inv_op };
    } else {
        return MarkedOp<D, NullInv, false, false, false, true>{std::forward<T>(f), {}};
    }
}


template <typename T>
consteval auto get_base_op() {
    if constexpr (requires { typename decay_t<T>::base_op; }) {
        return type_identity<typename decay_t<T>::base_op>{};
    } else {
        return type_identity<decay_t<T>>{};
    }
}

template <typename T>
using unwrap_marker_t = typename decltype(get_base_op<T>())::type;

template <typename T> concept IsCommutative = requires { requires decay_t<T>::is_comm; };
template <typename T> concept IsSingle = requires { requires decay_t<T>::is_single; };
template <typename T> concept IsInvertible  = requires { requires decay_t<T>::is_inv; };
template <typename T> concept IsEarlyExit   = requires { requires decay_t<T>::is_early_exit; };
template <typename T> concept IsHeavyWeight = requires { requires decay_t<T>::is_heavy; };

template <typename T> concept IsAbelian = IsInvertible<T> && IsCommutative<T>;
template <typename Op, typename Op2>
concept IsAbsorbing = (
    is_same_v<unwrap_marker_t<Op>, unwrap_marker_t<Op2>> &&
    (IsCommutative<Op> || IsCommutative<Op2>)
);

template <typename T>
requires IsInvertible<T>
[[gnu::always_inline]] 
constexpr auto extract_inverse(const T& wrap) {
    return wrap.inv_op;
}

template <typename F>
concept IsNotRawFunction = !is_function_v<remove_pointer_t<decay_t<F>>>;

template <typename F, typename... Args>
requires (!IsHeavyWeight<F>)
[[gnu::flatten, gnu::always_inline]]
constexpr decltype(auto) invoke_flat(F&& f, Args&&... args) {
    return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <typename F, typename... Args>
requires IsHeavyWeight<F>
[[gnu::always_inline]]
constexpr decltype(auto) invoke_flat(F&& f, Args&&... args) {
    return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <typename T, typename OriginalOp, typename... Args>
[[gnu::always_inline]]
constexpr auto convert_to_unary_op(OriginalOp&& orig_op, Args&&... args) {
    return [
        orig = std::forward<OriginalOp>(orig_op), 
        ...args = std::forward<Args>(args)
    ](T x) __attribute__((always_inline)) -> decltype(auto) {
        return invoke_flat(orig, x, args...);
    };
}

#define UNARY_OP(op) convert_to_unary_op<T>((op), std::forward<Args>(args)...)

enum class RangeType { ANY, PREFIX, SUFFIX, ALL };

struct reserve_t {};
constexpr reserve_t RESERVE_ONLY{};

// https://github.com/bspur
template <
    class T, auto upd = assignment{}, auto combine = plus{},
    auto e = nullptr, auto ok = missing_predicate{},
    bool N4 = false, bool support_empty_ranges = true
>
requires
    ValidCombine<decltype(combine), T> &&
    ValidIdentity<decltype(e), T>
class GenericSegTree {
    static_assert(
        !(N4 && is_same_v<decltype(e), nullptr_t>), 
        "when using 4N mode, you must explicitly provide the identity element e"
    );
private:
    static constexpr auto UPD = []() { 
        if constexpr (is_null_pointer_v<decltype(upd)>)      return assignment{};
        else                                                 return upd;
    }();
    static constexpr auto COMBINE_BASE = []() {
        if constexpr (is_null_pointer_v<decltype(combine)>)  return plus{};
        else                                                 return combine;
    }();
    using OC = decltype(COMBINE_BASE);

    // signature normalization
    static constexpr auto COMBINE = [](auto&& x, auto&& y, uint idx, uint szl, uint szr)
        __attribute__((always_inline)) -> decltype(auto) { 
        auto dispatch = [&](auto&&... tail) __attribute__((always_inline)) -> decltype(auto) {
            return invoke_flat(
                COMBINE_BASE, 
                std::forward<decltype(x)>(x), 
                std::forward<decltype(y)>(y), 
                std::forward<decltype(tail)>(tail)...
            );
        };
        if constexpr (is_invocable_r_v<T, OC, T, T>)                  return dispatch(); 
        else if constexpr (is_invocable_r_v<T, OC, T, T, uint>)       return dispatch(idx); 
        else if constexpr (is_invocable_r_v<T, OC, T, T, uint, uint>) return dispatch(szl, szr); 
        else                                                          return dispatch(idx, szl, szr); 
    };

    // https://gemini.google.com/app/5e9f9dc6ed77a6bd
    static constexpr T E() __attribute__((always_inline)) {
        if constexpr (is_null_pointer_v<decltype(e)>)        return T{}; // value-initialized
        else if constexpr (is_invocable_r_v<T, decltype(e)>) return e(); // function call
        else                                                 return e;   // raw value
    }
    
    static constexpr auto OK = []() {
        if constexpr (is_null_pointer_v<decltype(ok)>)       return missing_predicate{};
        else                                                 return ok;
    }();

    // circumventing pathological vector<bool>
    using StoreT = conditional_t<is_same_v<T, bool>, uint8_t, T>;

    uint n;
    uint size;
    uint log_;
    vector<StoreT, AlignedAllocator<StoreT>> seg;

    static constexpr uint calc_size(uint n) { return N4 ? bit_ceil(n) : n; }

    void init_(uint N) {
        n = N;
        size = calc_size(n);
        log_ = size ? __lg(size) : 0;
        seg.resize(size << 1);
    }

public:
    using enum RangeType;

    // INITIALIZERS
    void init() { init_(0); }
    void init(uint N) {
        init_(N);
        fill_n(seg.begin() + size, n, E());
        build_tree<true>();
    }

    void init(uint N, T x) {
        init_(N);
        fill_n(seg.begin() + size, n, x);
        build_tree();
    }
    template <ranges::input_range R>
    requires convertible_to<ranges::range_value_t<R>, T>
    void init(R&& range) {
        init_(ranges::size(range));
        ranges::copy(std::forward<R>(range), seg.begin() + size);
        build_tree();
    }

    // generator over indices
    template <class F>
    requires is_invocable_r_v<T, F, uint>
    void init(uint N, F gen) {
        init_(N);
        // __restrict guarantees + alignment guarantees
        auto* __restrict dest = assume_aligned<CACHE_LINE_SIZE>(seg.data());
        for (uint i = 0; i < n; ++i) {
            dest[size + i] = invoke_flat(gen, i);
        }
        build_tree();
    }

    // generator over values
    template <ranges::input_range R, class F>
    requires is_invocable_r_v<T, F, ranges::range_reference_t<R>>
    void init(R&& range, F gen) {
        init_(ranges::size(range));
        // __restrict guarantees + alignment guarantees
        auto* __restrict dest = assume_aligned<CACHE_LINE_SIZE>(seg.data());
        if constexpr (ranges::contiguous_range<R>) {
            const auto* src = ranges::data(range);
            for (uint i = 0; i < n; ++i) {
                dest[size + i] = invoke_flat(gen, src[i]);
            }
        } else {
            for (uint i = 0; auto&& val : range) {
                dest[size + i++] = invoke_flat(gen, val);
            }
        }
        build_tree();
    }

    // buffer provide
    template <typename F>
    requires is_invocable_v<F, T*>
    void init(uint N, F&& leaf_filler) {
        init_(N);
        StoreT* ptr = seg.data() + size;
        if constexpr (is_same_v<T, bool>) {
            // placement new (prevents strict aliasing issues)
            for (uint i = 0; i < n; i++) {
                new (ptr + i) bool; 
            }
        }
        invoke_flat(std::forward<F>(leaf_filler), reinterpret_cast<T*>(ptr));
        build_tree();
    }

    // CONSTRUCTORS
    explicit GenericSegTree() { init(0); }
    explicit GenericSegTree(uint N) { init(N); }
    explicit GenericSegTree(uint N, T x) { init(N, x); }
    template <class F>
    requires is_invocable_r_v<T, F, uint>
    explicit GenericSegTree(uint N, F gen) { init(N, gen); }
    template <ranges::input_range R, class F>
    requires is_invocable_r_v<T, F, ranges::range_reference_t<R>>
    explicit GenericSegTree(R&& range, F gen) { init(std::forward<R>(range), gen); }
    template <typename F>
    requires is_invocable_v<F, T*>
    explicit GenericSegTree(uint N, F&& leaf_filler) {
        init(N, std::forward<F>(leaf_filler));
    }
    template <ranges::input_range R>
    requires convertible_to<ranges::range_value_t<R>, T>
    explicit GenericSegTree(R&& range) { init(std::forward<R>(range)); }

    // capacity-only constructor
    // WARNING: MUST CALL INIT() BEFORE USE
    // GenericSegTree(n, RESERVE_ONLY)
    explicit GenericSegTree(uint n, reserve_t) {
        uint size = calc_size(n);
        seg.reserve(size << 1);
    }

    template <bool NoCombine = false>
    [[gnu::always_inline]]
    inline void build_tree() {
        auto update_node = [&](uint i, uint sh, uint sz) {
            seg[i] = NoCombine
                ? E()
                : COMBINE(seg[i<<1], seg[i<<1|1], (i << sh) + sz - size, sz, sz);
        };
        if constexpr (N4) {
            // only sets 2N + log(N) nodes
            uint l = size, r = size + n - 1, sz = 1, sh = 1;
            if (!(r&1)) seg[r+1] = E();
            for (l >>= 1, r >>= 1; r; l >>= 1, r >>= 1, sz <<= 1, sh++) {
                for (uint i = l; i <= r; i++) {
                    update_node(i, sh, sz);
                }
                if (!(r&1)) seg[r+1] = E();
            }
        } else {
            uint I = n, i = n, sz = 1, sh = 1;
            for (; I >>= 1; sz <<= 1, sh++) {
                while (i > I) {
                    i--;
                    update_node(i, sh, sz);
                }
            }
        }
    }

    template <typename... Args>
    requires is_invocable_r_v<T, decltype(UPD), T, Args...>
    [[gnu::always_inline]]
    inline void modify(uint i, Args&&... args) {
        modify_<decltype(UPD)>(i, std::move(UNARY_OP(UPD)));
    }

    template <auto update, typename... Args>
    requires is_invocable_r_v<T, decltype(update), T, Args...>
    [[gnu::always_inline]]
    inline void modify(uint i, Args&&... args) {
        modify_<decltype(update)>(i, std::move(UNARY_OP(update)));
    }

    template <class F, typename... Args>
    requires is_invocable_r_v<T, remove_cvref_t<F>, T, Args...> 
        && IsNotRawFunction<F>
    [[gnu::always_inline]]
    inline void modify(uint i, F&& update, Args&&... args) {
        modify_<remove_cvref_t<F>>(i, std::move(UNARY_OP(std::forward<F>(update))));
    }

    template <class OF, class F>
    [[gnu::always_inline]]
    inline void modify_(uint i, F update) {
        i += size;
        uint sz = 1, sh = 1;

        if constexpr (!IsEarlyExit<OF>) {
            if constexpr (IsAbsorbing<OF, OC> && is_invocable_v<OC, T, T>) {
                for (; i; i >>= 1) {
                    seg[i] = update(seg[i]);
                }
            } else if constexpr (IsAbelian<OC> && is_invocable_v<OC, T, T>) {
                constexpr auto inv_op = extract_inverse(COMBINE_BASE);
                T val = update(seg[i]);
                T delta = inv_op(val, seg[i]);
                seg[i] = val;
                while (i >>= 1) {
                    seg[i] = COMBINE(seg[i], delta, 0, 0, 0);
                }
            } else if constexpr (IsCommutative<OC>) {
                T val = update(seg[i]);
                seg[i] = val;
                T sib = seg[i ^ 1];
                for (; i >>= 1; sz <<= 1, sh += 1) {
                    val = COMBINE(val, sib, (i << sh) + sz - size, sz, sz);
                    seg[i] = val;
                    sib = seg[i ^ 1];
                }
            } else {
                seg[i] = update(seg[i]);
                for (; i >>= 1; sz <<= 1, sh += 1) {
                    seg[i] = COMBINE(seg[i<<1], seg[i<<1|1], (i << sh) + sz - size, sz, sz);
                }
            }
        } else { // EarlyExit
            if constexpr (IsAbsorbing<OF, OC> && is_invocable_v<OC, T, T>) {
                for (; i; i >>= 1) {
                    T val = update(seg[i]);
                    if (seg[i] == val) return;
                    seg[i] = val;
                }
            } else if constexpr (IsAbelian<OC> && is_invocable_v<OC, T, T>) {
                constexpr auto inv_op = extract_inverse(COMBINE_BASE);
                T val = update(seg[i]);
                if (seg[i] == val) return;
                seg[i] = val;
                T delta = inv_op(val, prev);
                while (i >>= 1) {
                    val = COMBINE(seg[i], delta, 0, 0, 0);
                    if (seg[i] == val) return;
                    seg[i] = val;
                }
            } else if constexpr (IsCommutative<OC>) {
                T val = update(seg[i]);
                if (seg[i] == val) return;
                seg[i] = val;
                T sib = seg[i ^ 1];
                for (; i >>= 1; sz <<= 1, sh += 1) {
                    val = COMBINE(val, sib, (i << sh) + sz - size, sz, sz);
                    if (seg[i] == val) return;
                    seg[i] = val;
                    sib = seg[i ^ 1];
                }
            } else {
                T val = update(seg[i]);
                if (seg[i] == val) return;
                seg[i] = val;
                for (; i >>= 1; sz <<= 1, sh += 1) {
                    val = COMBINE(seg[i<<1], seg[i<<1|1], (i << sh) + sz - size, sz, sz);
                    if (seg[i] == val) return;
                    seg[i] = val;
                }
            }
        }
    }

    #define GENERATE_QUERY_API(NAME, RT, L_VAL, R_VAL, ...) \
        [[gnu::always_inline]] \
        inline T NAME(__VA_OPT__(__VA_ARGS__)) const { \
            return query__<RT>(L_VAL, R_VAL); \
        }

    GENERATE_QUERY_API(query_all,    ALL,    0, n)
    GENERATE_QUERY_API(query_prefix, PREFIX, 0, r, uint r)
    GENERATE_QUERY_API(query_suffix, SUFFIX, l, n, uint l)
    GENERATE_QUERY_API(query,        ANY,    l, r, uint l, uint r)
    #undef GENERATE_QUERY_API

    template <RangeType RT>
    [[gnu::always_inline]]
    inline T query__(uint l, uint r) const {
        if (support_empty_ranges && is_empty_range<RT>(l, r)) return E();

        if constexpr (N4 && RT == ALL) {
            return seg[1];
        } else {
            uint hl, hr;
            prepare_bounds_and_mask<RT>(l, r, hl, hr);

            auto accumulate_right = [&]() __attribute__((always_inline)) {
                T ret = seg[(r >> countr_zero(hr)) - 1];
                // common subexpression elimination for hr & -hr
                for (uint R = r & r - 1; hr &= hr - 1; R -= hr & -hr) {
                    ret = COMBINE(seg[(r >> countr_zero(hr)) - 1], ret, R - size, hr & -hr, r - R);
                }
                return ret;
            };

            if (hl) {
                T ret = seg[(l >> countr_zero(hl)) + 1];
                uint L0 = l + 1, L = L0 + (hl & -hl);
                for (; hl &= hl - 1; L += hl & -hl) {
                    ret = COMBINE(ret, seg[(l >> countr_zero(hl)) + 1], L - size, L - L0, hl & -hl);
                }
                if (hr) {
                    ret = COMBINE(ret, accumulate_right(), L - size, L - L0, r - L);
                }
                return ret;
            }
            return accumulate_right();
        }
    }

    [[gnu::always_inline]]
    inline T query_point(uint i) const { return static_cast<T>(seg[i + size]); }
    [[gnu::always_inline]]
    inline T operator[] (uint i) const { return static_cast<T>(seg[i + size]); }

    // TODO: move these into the private domain
    static constexpr bool is_pow2 = has_single_bit(sizeof(StoreT));
    static constexpr uint max_depth = __lg(max<size_t>(2, (CACHE_LINE_SIZE + 1) / sizeof(StoreT) + 1));

    [[gnu::always_inline]]
    inline void prefetch_nodes_left(uint i, uint sh) const {
        __builtin_prefetch(&seg[i << sh]);
        if constexpr (!is_pow2) {
            __builtin_prefetch(reinterpret_cast<const byte*>(&seg[((i + 1) << sh) - 1]) - 1);
        }
    }

    // VRP + DCE will prune unreachable cases
    #define PREFETCH_HEAD(METHOD) \
        switch (cap) { \
            case 5: METHOD(i, 5); [[fallthrough]]; \
            case 4: METHOD(i, 4); [[fallthrough]]; \
            case 3: METHOD(i, 3); [[fallthrough]]; \
            case 2: METHOD(i, 2); \
        }

    #define DIG_TAIL(ACTION) \
        switch (height) { \
            case 5: ACTION; [[fallthrough]]; \
            case 4: ACTION; [[fallthrough]]; \
            case 3: ACTION; [[fallthrough]]; \
            case 2: ACTION; [[fallthrough]]; \
            case 1: ACTION; \
        }

    template <int phase, class F, class G>
    [[gnu::always_inline]]
    inline int dive_left_acc(uint i, uint height, T acc, uint cap, F okay, G comb, uint L0, uint L) const {
        if constexpr (phase <= 1) {
            cap = min(height, max_depth - 1u);
            PREFETCH_HEAD(prefetch_nodes_left);
        }
        #define DIG_LEFT_ACC() \
            do { \
                height--; \
                T ACC = comb(acc, seg[i <<= 1], L - size, L - L0, 1 << height); \
                if (!(okay(ACC))) { \
                    acc = ACC; \
                    i++; \
                    L += 1 << height; \
                } \
            } while (0)
        if constexpr (phase <= 2) {
            while (height > cap) {
                if constexpr (max_depth >= 2) prefetch_nodes_left(i, max_depth);
                DIG_LEFT_ACC();
            }
        }
        if constexpr (phase <= 3) {
            // Flow-Sensitive Constant Propagation will inline the
            // exact number of DIG_LEFT_ACC() needed in a dive_left_acc<3> call
            DIG_TAIL(DIG_LEFT_ACC());
        }
        #undef DIG_LEFT_ACC
        return i - size;
    }

    template <class F, class G>
    [[gnu::always_inline]]
    inline int dive_left(uint i, uint height, F okay, G comb) const {
        uint cap = min(height, max_depth - 1u);
        PREFETCH_HEAD(prefetch_nodes_left);
        #define DIG_LEFT(phase) \
            do { \
                height--; \
                T val = seg[i <<= 1]; \
                if (!(okay(val))) { \
                    uint L0 = i << height; \
                    uint L = (i + 1) << height; \
                    return dive_left_acc<(phase)>(i + 1, height, val, cap, okay, comb, L0, L); \
                } \
            } while (0)
        while (height > cap) {
            if constexpr (max_depth >= 2) prefetch_nodes_left(i, max_depth);
            DIG_LEFT(2);
        }
        DIG_TAIL(DIG_LEFT(3));
        #undef DIG_LEFT
        return i - size;
    }

    #define GENERATE_FIND_API(NAME, DISPATCHER, RT, L_VAL, R_VAL, ...) \
        template<typename... Args> \
        requires predicate<decltype(OK), T, Args...> \
        int NAME(__VA_OPT__(__VA_ARGS__,) Args&&... args) const { \
            return DISPATCHER<RT, decltype(OK)>(L_VAL, R_VAL, UNARY_OP(OK)); \
        } \
        template <auto okay, typename... Args> \
        requires predicate<decltype(okay), T, Args...> \
        int NAME(__VA_OPT__(__VA_ARGS__,) Args&&... args) const { \
            return DISPATCHER<RT, decltype(okay)>(L_VAL, R_VAL, UNARY_OP(okay)); \
        } \
        template <class F, typename... Args> \
        requires predicate<remove_cvref_t<F>, T, Args...> && IsNotRawFunction<F> \
        int NAME(__VA_OPT__(__VA_ARGS__,) F&& okay, Args&&... args) const { \
            return DISPATCHER<RT, remove_cvref_t<F>>(L_VAL, R_VAL, UNARY_OP(std::forward<F>(okay))); \
        }

    GENERATE_FIND_API(find_left_all,    find_left__,  ALL,    0, n)
    GENERATE_FIND_API(find_left_prefix, find_left__,  PREFIX, 0, r, uint r)
    GENERATE_FIND_API(find_left_suffix, find_left__,  SUFFIX, l, n, uint l)
    GENERATE_FIND_API(find_left,        find_left__,  ANY,    l, r, uint l, uint r)

    template <RangeType RT, class OF, class F>
    [[gnu::always_inline]]
    inline int find_left__(uint l, uint r, F okay) const {
        if constexpr (IsSingle<OF>) {
            return find_left__<RT>(l, r, okay, [](T, T y, uint, uint, uint) { return y; });
        } else {
            return find_left__<RT>(l, r, okay, COMBINE);
        }
    }

    template <RangeType RT>
    [[gnu::always_inline]]
    constexpr bool is_empty_range(uint l, uint r) const {
        if constexpr (RT == ALL) {
            return n == 0;
        } else if constexpr (RT == PREFIX) {
            return r == 0;
        } else if constexpr (RT == SUFFIX) {
            return l == n;
        } else { // (RT == ANY)
            return l == r;
        }
    }

    template <RangeType RT>
    [[gnu::always_inline]]
    constexpr void prepare_bounds_and_mask(uint& l, uint& r, uint& hl, uint& hr) const {
        if constexpr (N4) {
            if constexpr (RT == ANY) {
                l += size - 1, r += size;
                uint anc_mask = (1 << (bit_width(l ^ r) - 1)) - 1;
                hl = ~l & anc_mask, hr = r & anc_mask;
            } else if constexpr (RT == PREFIX) {
                hl = 0, hr = r;
                r += size;
            } else if constexpr (RT == SUFFIX) {
                // (l, r]
                l += size - 1, r += size - 1;
                // calculations needed to reduce working set of memory to 2N + log(N)
                uint anc_mask = (1 << (bit_width(l ^ r))) - 1;
                hl = ~l & anc_mask, hr = 0;
            } else { // RT == ALL
                static_assert(RT != ALL, "please use fast-tracks for N4 + ALL cases :)");
            }
        } else {
            if constexpr (RT == ALL) {
                l = n - 1, r = n << 1;
                uint k = 1u << bit_width(n);
                hl = k - n, hr = r - k;
            } else { // RT == PREFIX / SUFFIX / ANY
                l += n - 1, r += n;
                uint anc_mask = (1 << (bit_width(l ^ r) - 1)) - 1;
                hl = ~l & anc_mask, hr = r & anc_mask;
            }
        }
    }

    template <RangeType RT, class F, class G>
    [[gnu::always_inline]]
    inline int find_left__(uint l, uint r, F okay, G comb) const {
        if (support_empty_ranges && is_empty_range<RT>(l, r)) return -1;

        // fast track for 4N + ALL
        if constexpr (N4 && RT == ALL) {
            if (okay(seg[1])) {
                return dive_left(1, log_, okay, comb);
            }
        } else {
            uint hl, hr;
            prepare_bounds_and_mask<RT>(l, r, hl, hr);

            if (hl) {
                // first step of ascent
                uint i = countr_zero(hl);
                T val = seg[(l >> i) + 1];
                if (okay(val)) {
                    return dive_left((l >> i) + 1, i, okay, comb);
                } else {
                    return find_left_ascent(l, r, hl, hr, val, okay, comb);
                }
            }
            if (hr) {
                // first step of descent
                int pcnt = _mm_popcnt_u32(hr);
                uint sh = 1 << (pcnt - 1);
                uint m = _pdep_u32(sh, hr);
                uint i = countr_zero(m);

                T val = seg[(r >> i) - 1];
                if (okay(val)) {
                    return dive_left((r >> i) - 1, i, okay, comb);
                } else {
                    return find_left_descent(r, hr, sh, val, okay, comb, r - hr, r - hr + m);
                }
            }
        }
        return -1;
    }

    // NOTE: aggressive inlining is not likely to cause slowdown
    // via. instruction bloat (spilling out of uop cache)
    // because segment tree sizes are typically large enough
    // such that they sit mostly in L2 / L3 cache,
    // and L1i cache is much faster than those
    template <class F, class G>
    [[gnu::always_inline]]
    inline int find_left_ascent(uint l, uint r, uint hl, uint hr, T acc, F okay, G comb) const {
        uint L0 = l + 1, L = L0 + (hl & -hl);
        for (; hl &= hl - 1; L += hl & -hl) {
            uint i = countr_zero(hl);
            T ACC = comb(acc, seg[(l >> i) + 1], L - size, L - L0, hl & -hl);
            if (okay(ACC)) {
                return dive_left_acc<1>((l >> i) + 1, i, acc, 0, okay, comb, L0, L);
            }
            acc = ACC;
        }
        int pcnt = _mm_popcnt_u32(hr);
        return find_left_descent(r, hr, 1 << pcnt, acc, okay, comb, L0, L);
    }

    template <class F, class G>
    [[gnu::always_inline]]
    inline int find_left_descent(uint r, uint hr, uint sh, T acc, F okay, G comb, uint L0, uint L) const {
        while(sh >>= 1) {
            uint m = _pdep_u32(sh, hr);
            uint i = countr_zero(m);
            T ACC = comb(acc, seg[(r >> i) - 1], L - size, L - L0, m);
            if (okay(ACC)) {
                return dive_left_acc<1>((r >> i) - 1, i, acc, 0, okay, comb, L0, L);
            }
            L += m;
        }
        return -1;
    }

    [[gnu::always_inline]]
    inline void prefetch_nodes_right(uint i, uint sh) const {
        __builtin_prefetch(&seg[(i << sh) + 1]);
        if constexpr (!is_pow2) {
            __builtin_prefetch(reinterpret_cast<const byte*>(&seg[(i + 1) << sh]) - 1);
        }
    }

    template <int phase, class F, class G>
    [[gnu::always_inline]]
    inline int dive_right_acc(uint i, uint height, T acc, uint cap, F okay, G comb, uint R0, uint R) const {
        if constexpr (phase <= 1) {
            cap = min(height, max_depth - 1u);
            PREFETCH_HEAD(prefetch_nodes_right);
        }
        #define DIG_RIGHT_ACC() \
            do { \
                height--; \
                i = (i << 1) + 1; \
                T ACC = comb(seg[i], acc, R - size, 1 << height, R0 - R); \
                if (!(okay(ACC))) { \
                    acc = ACC; \
                    i--; \
                    R -= 1 << height; \
                } \
            } while (0)
        if constexpr (phase <= 2) {
            while (height > cap) {
                if constexpr (max_depth >= 2) prefetch_nodes_right(i, max_depth);
                DIG_RIGHT_ACC();
            }
        }
        if constexpr (phase <= 3) {
            // Flow-Sensitive Constant Propagation will inline
            // the exact number of DIG_LEFT_ACC() needed in a dive_left_acc<3> call
            DIG_TAIL(DIG_RIGHT_ACC());
        }
        #undef DIG_RIGHT_ACC
        return i - size;
    }

    template <class F, class G>
    [[gnu::always_inline]]
    inline int dive_right(uint i, uint height, F okay, G comb) const {
        uint cap = min(height, max_depth - 1u);
        PREFETCH_HEAD(prefetch_nodes_right);
        #define DIG_RIGHT(phase) \
            do { \
                height--; \
                i = (i << 1) + 1; \
                T val = seg[i]; \
                if (!(okay(val))) { \
                    uint R0 = (i + 1) << height; \
                    uint R = i << height; \
                    return dive_right_acc<(phase)>(i - 1, height, val, cap, okay, comb, R0, R); \
                } \
            } while (0)
        while (height > cap) {
            if constexpr (max_depth >= 2) prefetch_nodes_right(i, max_depth);
            DIG_RIGHT(2);
        }
        DIG_TAIL(DIG_RIGHT(3));
        #undef DIG_RIGHT
        return i - size;
    }

    GENERATE_FIND_API(find_right_all,    find_right__, ALL,    0, n)
    GENERATE_FIND_API(find_right_prefix, find_right__, PREFIX, 0, r, uint r)
    GENERATE_FIND_API(find_right_suffix, find_right__, SUFFIX, l, n, uint l)
    GENERATE_FIND_API(find_right,        find_right__, ANY,    l, r, uint l, uint r)
    #undef GENERATE_FIND_API
    #undef PREFETCH_HEAD
    #undef DIG_TAIL

    template <RangeType RT, class OF, class F>
    [[gnu::always_inline]]
    inline int find_right__(uint l, uint r, F okay) const {
        if constexpr (IsSingle<OF>) {
            return find_right__<RT>(l, r, okay, [](T x, T, uint, uint, uint) { return x; });
        } else {
            return find_right__<RT>(l, r, okay, COMBINE);
        }
    }

    template <class F, class G>
    [[gnu::always_inline]]
    inline int find_right_ascent(uint l, uint r, uint hl, uint hr, T acc, F okay, G comb) const {
        uint R0 = r;
        uint R = R0 + (hr & -hr);

        for (; hr &= hr - 1; R -= hr & -hr) {
            uint i = countr_zero(hr);
            T ACC = comb(seg[(r >> i) - 1], acc, R - size, hr & -hr, R0 - R);
            if (okay(ACC)) {
                return dive_right_acc<1>((r >> i) - 1, i, acc, 0, okay, comb, R0, R);
            }
            acc = ACC;
        }

        int pcnt = _mm_popcnt_u32(hl);
        return find_right_descent(l, hl, 1 << pcnt, acc, okay, comb, R0, R);
    }

    template <class F, class G>
    [[gnu::always_inline]]
    inline int find_right_descent(uint l, uint hl, uint sh, T acc, F okay, G comb, uint R0, uint R) const {
        while(sh >>= 1) {
            uint m = _pdep_u32(sh, hl);
            uint i = countr_zero(m);

            T ACC = comb(seg[(l >> i) + 1], acc, R - size, m, R0 - R);
            if (okay(ACC)) {
                return dive_right_acc<1>((l >> i) + 1, i, acc, 0, okay, comb, R0, R);
            }
            R -= m;
        }
        return -1;
    }

    template <RangeType RT, class F, class G>
    [[gnu::always_inline]]
    inline int find_right__(uint l, uint r, F okay, G comb) const {
        if (support_empty_ranges && is_empty_range<RT>(l, r)) return -1;

        if constexpr (N4 && (RT == ALL || RT == SUFFIX)) {
            assert(
                !okay(E()) &&
                "pred(e) must fail for 4N mode to work properly for "
                "find_right_suffix / find_right_all. "
                "Please switch to 2N mode."
            );
        }

        // fast track for 4N + ALL
        if constexpr (N4 && RT == ALL) {
            if (okay(seg[1])) {
                return dive_right(1, log_, okay, comb);
            }
        } else {
            uint hl, hr;
            prepare_bounds_and_mask<RT>(l, r, hl, hr);

            if (hr) {
                uint i = countr_zero(hr);
                T val = seg[(r >> i) - 1];

                if (okay(val)) {
                    return dive_right((r >> i) - 1, i, okay, comb);
                } else {
                    return find_right_ascent(l, r, hl, hr, val, okay, comb);
                }
            }
            if (hl) {
                int pcnt = _mm_popcnt_u32(hl);
                uint sh = 1 << (pcnt - 1);
                uint m = _pdep_u32(sh, hl);
                uint i = countr_zero(m);

                T val = seg[(l >> i) + 1];
                if (okay(val)) {
                    return dive_right((l >> i) + 1, i, okay, comb);
                } else {
                    return find_right_descent(l, hl, sh, val, okay, comb, r - hr, r - hr - m);
                }
            }
        }
        return -1;
    }
};



#ifdef __clang__
    #pragma clang attribute pop
#else
    #pragma GCC pop_options
#endif
