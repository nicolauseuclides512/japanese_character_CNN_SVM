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
    Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/train/Hiragana', data_trains, label_trains)
    Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/test/Hiragana', data_tests, label_tests)
    Process.katakanaPreprocess(main_directory, main_directory + 'data_image/train/Katakana', data_trains, label_trains)
    Process.katakanaPreprocess(main_directory, main_directory + 'data_image/train/Katakana', data_tests, label_tests)
    print("done")
    NNet.convnetModel(data_trains, label_trains, 168, 168)


if __name__ == "__main__":
    main()
