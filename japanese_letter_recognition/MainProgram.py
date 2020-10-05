import japanese_letter_recognition.Preprocess as Process
import japanese_letter_recognition.NeuralNetwork as NNet


def main():
    data_trains = []
    data_tests = []
    label_trains = []
    label_tests = []
    main_directory = "/home/nicolaus/PycharmProjects/thesis_program/"
    print("Japanese Pattern Recognition")
    Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/train/Hiragana', data_trains, label_trains)
    Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/test/Hiragana', data_tests, label_tests)
    Process.katakanaPreprocess(main_directory, main_directory + 'data_image/train/Katakana', data_trains, label_trains)
    Process.katakanaPreprocess(main_directory, main_directory + 'data_image/train/Katakana', data_tests, label_tests)
    print("done")
    data_trains, data_vals, label_trains, label_vals = NNet.splitData(data_trains, label_trains)
    features_train = NNet.convnetModel(data_trains, label_trains, 168, 168).predict
    print("Fitur = "+features_train)


if __name__ == "__main__":
    main()
