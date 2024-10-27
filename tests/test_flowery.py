import unittest
from garden import flowery

class TestFlowery(unittest.TestCase):
    def test_suppress(self):
        class Example:
            @flowery(verbose=False)
            def method(self):
                print("hidden")

        with self.assertLogs() as log:
            Example().method()
            self.assertEqual(len(log.output), 0)

if __name__ == '__main__':
    unittest.main()

