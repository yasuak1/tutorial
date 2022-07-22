require 'minitest/autorun'

def fizz_buzz(n)
    if n % 15 == 0
        'Fizz Buzz'
    elsif n % 3 == 0
        'Fizz'
    elsif n % 5 == 0
        'Buzz'
    else
        n.to_s
    end
end

class FizzBuzzTest < Minitest::Test
    def test_fizz_buzz
        assert '1' == fizz_buzz(1)
        assert '2' == fizz_buzz(2)
        assert 'Fizz' == fizz_buzz(3)
    end
end

#n = gets.chomp.to_i
#puts fizz_buzz(n)