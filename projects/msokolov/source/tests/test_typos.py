import unittest
import source.typos.hirschberg as rust_hirshberg


def dictionary():
    return ["mouse", "horse", "python"]


class TestTypos(unittest.TestCase):
    def test_fix_mouse_typo(self):
        dict = dictionary()
        answer = rust_hirshberg.try_correct(dict, "mpuse")
        self.assertEqual("mouse", answer)

    def test_fix_horse_typo(self):
        dict = dictionary()
        answer = rust_hirshberg.try_correct(dict, "horsi")
        self.assertEqual("horse", answer)

    def test_fix_python_typo(self):
        dict = dictionary()
        answer = rust_hirshberg.try_correct(dict, "puthon")
        self.assertEqual("python", answer)



if __name__ == '__main__':
    unittest.main()
