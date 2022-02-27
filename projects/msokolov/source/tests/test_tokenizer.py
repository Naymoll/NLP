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

    def test_time_6(self):
        time = self.tokenize_one('12:24 p.m.')
        self.assertEqual(time, '12:24 p.m.')

    def test_phone_number_1(self):
        phone = self.tokenize_one('(800)-100-2000')
        self.assertEqual(phone, '(800)-100-2000')

    def test_phone_number_2(self):
        phone = self.tokenize_one('800-100-2000')
        self.assertEqual(phone, '800-100-2000')

    def test_phone_number_3(self):
        phone = self.tokenize_one('800 100-2000')
        self.assertEqual(phone, '800 100-2000')

    def test_phone_number_4(self):
        phone = self.tokenize_one('(800) 100-2000')
        self.assertEqual(phone, '(800) 100-2000')

    def test_word_1(self):
        word = self.tokenize_one('Some')
        self.assertEqual(word, 'Some')

    def test_word_2(self):
        word = self.tokenize_one("aren't")
        self.assertEqual(word, "aren't")

    def test_date_1(self):
        date = self.tokenize_one("10.5.2003")
        self.assertEqual(date, "10.5.2003")

    def test_date_2(self):
        date = self.tokenize_one("9-12-2003")
        self.assertEqual(date, "9-12-2003")

    def test_date_3(self):
        date = self.tokenize_one("2005-11-23")
        self.assertEqual(date, "2005-11-23")

    def test_ip(self):
        ip = self.tokenize_one("127.0.0.1")
        self.assertEqual(ip, "127.0.0.1")

    def test_abbr(self):
        abbr = self.tokenize_one("U.S.A.")
        self.assertEqual(abbr, "U.S.A.")

    def test_url_1(self):
        url = r"http://www.investor.reuters.com/FullQuote.aspx?ticker=HD.N"
        url_token_text = self.tokenize_one(url)
        self.assertEqual(url_token_text, url)

    def test_url_2(self):
        url = r"www.craigslist.org"
        url_token_text = self.tokenize_one(url)
        self.assertEqual(url_token_text, url)

    def test_url_3(self):
        url = r"washingtonpost.com"
        url_token_text = self.tokenize_one(url)
        self.assertEqual(url_token_text, url)

    def test_meta_1(self):
        meta = r"&lt;A HREF=""http://www.investor.reuters.com/FullQuote.aspx?ticker=NT.TO target=/stocks/quickinfo/fullquote""&gt;"
        meta_token_text = self.tokenize_one(meta)
        self.assertEqual(meta_token_text, meta)

    def test_meta_2(self):
        meta = self.tokenize_one("#36;")
        self.assertEqual(meta, "#36;")


if __name__ == '__main__':
    unittest.main()
