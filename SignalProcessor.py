import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import welch, csd
from scipy.fft import fft
from scipy.signal import butter, lfilter

# 信号処理を行うクラス
class SignalProcessor:
    def __init__(self, input_data, output_data, fs, cutoff_freq):
        self.input_data = input_data  # 入力データ
        self.output_data = output_data  # 出力データ
        self.fs = fs  # サンプリング周波数
        self.cutoff_freq = cutoff_freq # カットオフ周波数 (Hz)

    def plot_fft_spectrum(self, signal, sampling_rate=2000):
        # FFTスペクトルをプロットする関数
        N = len(signal)  # サンプル点の数

        # FFTと周波数領域
        yf = fft(signal)
        xf = np.linspace(0.0, sampling_rate/2, N//2)

        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.title("FFT Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def apply_low_pass_filter(self, signal, cutoff_frequency, sampling_rate=2000):
        # ローパスフィルタを適用する関数
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist

        # Butterworthフィルタを設計
        b, a = butter(N=1, Wn=normal_cutoff, btype='low', analog=False)

        # フィルタを適用
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def compute_power_spectrum(self, data):
        # パワースペクトルを計算する関数
        f, Pxx = welch(data, fs=self.fs)
        return f, Pxx

    def compute_cross_spectrum(self):
        # クロススペクトルを計算する関数
        f, Pxy = csd(self.input_data, self.output_data, fs=self.fs)
        return f, Pxy

    def compute_frequency_response_function(self):
        # 周波数応答関数を計算する関数
        f, Pxx = self.compute_power_spectrum(self.input_data)
        _, Pxy = self.compute_cross_spectrum()
        H = Pxy / Pxx
        return f, H

    def plot_bode(self, f, H):
        # ボード線図（ゲインと位相のグラフ）を描画する関数
        gain = 20 * np.log10(np.abs(H))  # ゲインをデシベル単位で計算
        phase = np.angle(H, deg=True)  # 位相を度単位で計算

        self.plot_fft_spectrum(gain)
        self.plot_fft_spectrum(phase)

        gain = self.apply_low_pass_filter(gain, self.cutoff_freq)
        phase = self.apply_low_pass_filter(phase, self.cutoff_freq)

        # カスタムフォーマッタの定義
        def custom_formatter(x, pos):
            if x >= 1000:
                return '{:.0f}k'.format(x / 1000)
            else:
                return '{:.0f}'.format(x)

        # ゲイン線図を描画
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.semilogx(f, gain, base=2)
        plt.title('Bode Plot')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        # 位相線図を描画
        plt.subplot(2, 1, 2)
        plt.semilogx(f, phase, base=2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [degrees]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()
        plt.show()

    def find_flat_gain(self, f, H):
        # 平坦な部分のゲインを見つける関数
        gain = 20 * np.log10(np.abs(H))
        gain = self.apply_low_pass_filter(gain, self.cutoff_freq)
        # 平坦な部分を特定するための簡易アルゴリズム（例えば標準偏差が小さい区間）
        std_gain = np.convolve(gain, np.ones(10)/10, mode='valid')  # 移動平均に基づく標準偏差
        flat_idx = np.argmin(np.abs(std_gain))  # 最も平坦な部分
        return f[flat_idx], gain[flat_idx]

    def find_peak_gains(self, f, H):
        # 山の頂点のゲインを見つける関数
        gain = 20 * np.log10(np.abs(H))
        gain = self.apply_low_pass_filter(gain, self.cutoff_freq)
        peaks, _ = find_peaks(gain, height=0)  # 山の頂点を見つける
        # ゲインが高い順に並べ替え
        peak_gains = sorted(zip(f[peaks], gain[peaks]), key=lambda x: x[1], reverse=True)
        return peak_gains[:3]  # 最大3つの山

    def gain_difference(self, f, H):
        # 平坦な部分と山の頂点のゲインの差を計算する関数
        flat_freq, flat_gain = self.find_flat_gain(f, H)
        peak_gains = self.find_peak_gains(f, H)
        differences = [(pf, pg - flat_gain) for pf, pg in peak_gains]
        return differences

    def write_results_to_file(self, filepath, f, frequency_response):
        with open(filepath, 'w') as file:
            # 平坦なゲインの値を書き出す
            flat_freq, flat_gain = self.find_flat_gain(f, frequency_response)
            file.write(f"平坦なゲインの周波数: {flat_freq:.2f} Hz, ゲイン: {flat_gain:.2f} dB\n")

            # 山の頂点のゲインの値を書き出す
            peak_gains = self.find_peak_gains(f, frequency_response)
            file.write("山の頂点のゲインの値（上位3つ）:\n")
            for freq, gain in peak_gains:
                file.write(f"  周波数: {freq:.2f} Hz, ゲイン: {gain:.2f} dB\n")

            # ゲインの差を書き出す
            gain_diff = self.gain_difference(f, frequency_response)
            file.write("ゲインの差（平坦な部分と山の頂点）:\n")
            for freq, diff in gain_diff:
                file.write(f"  周波数: {freq:.2f} Hz, ゲインの差: {diff:.2f} dB\n")

'''
# サンプルデータでの使用例（実際のオーディオデータに置き換えてください）
input_data = np.random.randn(1000)  # ダミー入力データ
output_data = np.random.randn(1000)  # ダミー出力データ
sampling_rate = 48000  # サンプリングレートの例

# SignalProcessorのインスタンスを作成
signal_processor = SignalProcessor(input_data, output_data, sampling_rate)

# 入力のパワースペクトルを計算
f, power_spectrum = signal_processor.compute_power_spectrum(input_data)

# 入力と出力のクロススペクトルを計算
f, cross_spectrum = signal_processor.compute_cross_spectrum()

# 周波数応答関数を計算
f, frequency_response = signal_processor.compute_frequency_response_function()

# デモンストレーション用に最初の数値を表示
(f[:5], power_spectrum[:5]), (f[:5], cross_spectrum[:5]), (f[:5], frequency_response[:5])

# ボード線図を描画
signal_processor.plot_bode(f, frequency_response)

# 平坦なゲインの値を取得
flat_freq, flat_gain = signal_processor.find_flat_gain(f, frequency_response)

# 山の頂点のゲインの値を取得
peak_gains = signal_processor.find_peak_gains(f, frequency_response)

# ゲインの差を取得
gain_diff = signal_processor.gain_difference(f, frequency_response)

# 例としてダミー信号を使用します
# デモンストレーション用のダミー信号を作成します
np.random.seed(0)  # 乱数のシードを0に設定して、結果を再現可能にします
dummy_signal = np.random.normal(0, 1, 2000)  # 平均0、標準偏差1の正規分布に従う2000個の乱数を生成して、例の信号とします

# ダミー信号のFFTスペクトラムをプロットします
plot_fft_spectrum(dummy_signal)  # FFTスペクトラムを表示する関数を使用

# 例としてダミー信号とカットオフ周波数を使用します
cutoff_freq = 500  # カットオフ周波数をHz単位で設定します

# ダミー信号にLPF（ローパスフィルター）を適用します
filtered_signal = apply_low_pass_filter(dummy_signal, cutoff_freq)  # LPFを適用した信号を取得

# 元の信号とフィルター処理された信号を比較してプロットします
plt.figure(figsize=(12, 6))  # 図のサイズを設定
plt.plot(dummy_signal, label="Original Signal")  # 元の信号をプロット
plt.plot(filtered_signal, label="Filtered Signal", alpha=0.7)  # フィルター処理された信号をプロット、透明度を0.7に設定
plt.title("Original vs. Low-Pass Filtered Signal")  # タイトルを設定
plt.xlabel("Sample Number")  # X軸のラベルを設定
plt.ylabel("Amplitude")  # Y軸のラベルを設定
plt.legend()  # 凡例を表示
plt.grid()  # グリッド線を表示
plt.show()  # 図を表示
'''
