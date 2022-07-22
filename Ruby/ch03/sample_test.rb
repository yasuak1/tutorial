require 'minitest/autorun'

class SampleTest < Minitest::Test
    def test_sample
        assert_equal 'RUBY', 'ruby'.upcase
        assert_equal 'RUBY', 'ruby'.capitalize #SampleTest#test_sample [sample_test.rb:5]:
        assert_equal 'RUBY', NIL.upcase #NameError: uninitialized constant SampleTest::NIL
        #assert 'RUBY' == 'ruby'.upcase
        #refute 'RUBY' != 'ruby'.upcase
    end
end