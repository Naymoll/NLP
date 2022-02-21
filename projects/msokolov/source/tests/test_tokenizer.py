import unittest

from source.tokenizer import Tokenizer

tokenizer = Tokenizer()


class TestTokenizer(unittest.TestCase):
    def tokenize_one(self, text: str):
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 1)
        return tokens[0].text

    def test_number(self):
        number = self.tokenize_one('123')
        self.assertEqual(number, '123')

    def test_incorrect_number(self):
        tokens = tokenizer.tokenize('123:2434')
        self.assertEqual(len(tokens), 3)

    def test_float_format_1(self):
        number = self.tokenize_one('123,345')
        self.assertEqual(number, '123,345')

    def test_float_format_2(self):
        number = self.tokenize_one('123.345')
        self.assertEqual(number, '123.345')

    def test_time_1(self):
        time = self.tokenize_one('12:24')
        self.assertEqual(time, '12:24')

    def test_time_2(self):
        time = self.tokenize_one('12:24pm')
        self.assertEqual(time, '12:24pm')

    def test_time_3(self):
        time = self.tokenize_one('12:24 pm')
        self.assertEqual(time, '12:24 pm')

    def test_time_4(self):
        time = self.tokenize_one('12 pm')
        self.assertEqual(time, '12 pm')

    def test_time_5(self):
        time = self.tokenize_one('12:24p.m.')
        self.assertEqual(time, '12:24p.m.')

    def test_phone_number_1(self):
        time = self.tokenize_one('(800)-100-2000')
        self.assertEqual(time, '(800)-100-2000')

    def test_phone_number_2(self):
        time = self.tokenize_one('800-100-2000')
        self.assertEqual(time, '800-100-2000')

    def test_phone_number_3(self):
        time = self.tokenize_one('800 100-2000')
        self.assertEqual(time, '800 100-2000')

    def test_phone_number_4(self):
        time = self.tokenize_one('(800) 100-2000')
        self.assertEqual(time, '(800) 100-2000')

    def test_word_1(self):
        time = self.tokenize_one('Some')
        self.assertEqual(time, 'Some')

    def test_word_2(self):
        time = self.tokenize_one("aren't")
        self.assertEqual(time, "aren't")


if __name__ == '__main__':
    unittest.main()
