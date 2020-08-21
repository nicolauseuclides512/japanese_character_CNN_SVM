import japanese_letter_recognition.Preprocess as Process
import japanese_letter_recognition.NeuralNetwork as NNet


def main():
    data_with_label = []
    data_trains = []
    data_tests = []
    data_vals = []
    label_trains = []
    label_tests = []
    label_vals = []
    main_directory = "/home/nicolaus/PycharmProjects/thesis_program/"
    print("Japanese Pattern Recognition")
    Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/Hiragana', data_trains, data_tests,
                               data_vals, label_trains, label_tests, label_vals)
    Process.katakanaPreprocess(main_directory, main_directory + 'data_image/Katakana', data_trains, data_tests,
                               data_vals, label_trains, label_tests, label_vals)
    print("done")

if __name__ == "__main__":
    main()
