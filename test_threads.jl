function test_threads(a)

    @threads for i = 1:length(a)
        a[i] = threadid()
    end

end
function test(a)
    for i = 1:length(a)
        a[i] = i
    end

end
if true
    f(n) = begin
        a = zeros(10^n)
        println("Threads")
        display(@benchmark test_threads($a))
        println("Non-Threads")
        display(@benchmark test($a))
    end
    println("Compile")
    f(1)
    println("10^4 elements")
    f(4)
    println("10^5 elements")
    f(5)
end
