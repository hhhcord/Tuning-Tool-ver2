from AudioLoader import AudioLoader  # AudioLoaderクラスをインポート
from SignalProcessor import SignalProcessor  # SignalProcessorクラスをインポート

# AudioLoaderのインスタンスを作成
audio_loader = AudioLoader()

# load_audioメソッドを使用して、30秒間の入力オーディオデータを読み込む
input_data, sampling_rate = audio_loader.load_audio(30)

# 同様に、30秒間の出力オーディオデータを読み込む
output_data, sampling_rate = audio_loader.load_audio(30)

cutoff_freq = 80  # カットオフ周波数 (Hz)

# SignalProcessorのインスタンスを作成。入力データ、出力データ、サンプリングレートを渡す
signal_processor = SignalProcessor(input_data, output_data, sampling_rate, cutoff_freq)

# 周波数応答関数を計算する
f, frequency_response = signal_processor.compute_frequency_response_function()

# 結果をテキストファイルに書き出すためのメソッドを呼び出す
# "results.txt"には、計算された周波数応答関数のデータが保存される
signal_processor.write_results_to_file("results.txt", f, frequency_response)

# ボード線図を描画する
# ボード線図は、システムの周波数応答を示すグラフで、ゲインと位相の変化を視覚的に表示する
signal_processor.plot_bode(f, frequency_response)
