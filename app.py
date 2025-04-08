from cProfile import label
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import fft

from config import Config
from analysis import FileAnalyzer

class AudioApp:

    filepath: str
    analyzer: FileAnalyzer

    def __init__(self, root):
        self.root = root
        self.root.title("Analiza sygnału audio")

        window_height = Config.WINDOW_HEIGHT
        window_width = Config.WINDOW_WIDTH
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        menu = tk.Menu(root)
        
        file_menu = tk.Menu(menu)
        
        sub_menu = tk.Menu(file_menu, tearoff=0)
        sub_menu.add_command(label='Parametry ramek', command=lambda: self.save_file(clip=False))
        sub_menu.add_command(label='Parametry klipu', command=lambda: self.save_file(clip=True))
        file_menu.add_cascade(
            label="Eksportuj",
            menu=sub_menu
        )
        menu.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Wyjście", command=self.on_closing)

        menu.add_command(label="Ustawienia", command=self.show_settings)

        root.config(menu=menu)

        self.filepath = None

        self.create_widgets()
        self.enable_enter_for_buttons()
    
    def enable_enter_for_buttons(self):
        def trigger_focused_button(event):
            widget = self.root.focus_get()
            if isinstance(widget, tk.Button):
                widget.invoke()

        self.root.bind('<Return>', trigger_focused_button)

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def create_widgets(self):
        
        # file
        self.file_frame = tk.Frame(self.root)
        self.file_frame.pack()

        self.open_button = tk.Button(self.file_frame, text="Wczytaj plik WAV", command=self.open_file)
        self.open_button.pack(pady=(20, 10))

        self.file_label = tk.Label(self.file_frame, text="Wybrany plik: brak")
        self.file_label.pack()

        # analysis
        self.analysis_frame = tk.Frame(self.root)
        self.analysis_frame.pack(pady=(30, 0))

        # frame analyis
        self.frame_analysis_frame = tk.Frame(self.analysis_frame, relief=tk.RIDGE, borderwidth=2, border=1)
        self.frame_analysis_frame.pack(padx=20, side=tk.LEFT, fill='y')

        self.frame_label = tk.Label(self.frame_analysis_frame, text="Podstawowe wykresy")
        self.frame_label.pack(pady=5, padx=10)

        self.volume_button = tk.Button(self.frame_analysis_frame, text="Głośność", command=self.plot_volume, state=tk.DISABLED)
        self.volume_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')
        
        self.fc_button = tk.Button(self.frame_analysis_frame, text="Centroidy", command=self.plot_fc, state=tk.DISABLED)
        self.fc_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.sfm_button = tk.Button(self.frame_analysis_frame, text="SFM", command=self.plot_sfm, state=tk.DISABLED)
        self.sfm_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.f0_button = tk.Button(self.frame_analysis_frame, text="F0 (cepstrum)", command=self.plot_f0, state=tk.DISABLED)
        self.f0_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        # bands analysis
        self.bands_analysis_frame = tk.Frame(self.analysis_frame, relief=tk.RIDGE, borderwidth=2, border=1)
        self.bands_analysis_frame.pack(padx=20, side=tk.LEFT, fill='y')

        self.bands_label = tk.Label(self.bands_analysis_frame, text="Analiza pasm")
        self.bands_label.pack(pady=5, padx=10)

        self.ersb_button = tk.Button(self.bands_analysis_frame, text="ERSB (pasma)", command=self.plot_ersb, state=tk.DISABLED)
        self.ersb_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.bandwidth_button = tk.Button(self.bands_analysis_frame, text="Efektywne pasmo", command=self.plot_bandwidth, state=tk.DISABLED)
        self.bandwidth_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.sfmb_button = tk.Button(self.bands_analysis_frame, text="SFM (pasma)", command=self.plot_sfm_bands, state=tk.DISABLED)
        self.sfmb_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.scfb_button = tk.Button(self.bands_analysis_frame, text="SCF (pasma)", command=self.plot_scf_bands, state=tk.DISABLED)
        self.scfb_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')


        # clip analysis
        self.clip_analysis_frame = tk.Frame(self.analysis_frame, relief=tk.RIDGE, borderwidth=10, border=1)
        self.clip_analysis_frame.pack(padx=20, side=tk.RIGHT, fill='y')

        self.clip_label = tk.Label(self.clip_analysis_frame, text="Inne analizy")
        self.clip_label.pack(pady=5, padx=10)

        self.windows_button = tk.Button(self.clip_analysis_frame, text="Funkcje okna", command=self.window_analysis_window, state=tk.DISABLED)
        self.windows_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.spect_button = tk.Button(self.clip_analysis_frame, text="Spektrogram", command=self.spectrogram_window, state=tk.DISABLED)
        self.spect_button.pack(pady=5, padx=10, side=tk.TOP, fill='x') 

        # self.parameters_button = tk.Button(self.clip_analysis_frame, text="Parametry klipu", command=self.show_parameters, state=tk.DISABLED)
        # self.parameters_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        # canvas

        self.canvas_label = tk.Label(root, text="Przebieg czasowy pliku:")
        self.canvas_label.pack(anchor='w', pady=(30,0), padx=20)

        self.canvas_frame = tk.Frame(root, relief='solid', border=1, background='white')
        self.canvas_frame.pack(pady=(10, 20), padx=20)

        self.fig = plt.figure(constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(pady=10, padx=10)

    
    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        self.filepath = filepath
        try:
            self.analyzer = FileAnalyzer(filepath)
        except Exception as e:
            print("Cannot load file:", e)
            self.file_label.configure(text=f"Wybrany plik: Nie udało się wczytać wybranego pliku")
            return
        self.file_label.configure(text=f"Wybrany plik: {filepath.split('/')[-1]}")
        self.filepath = filepath
        self.activate_buttons()
        self.draw_waveform()
    
    def activate_buttons(self):
        frames = [self.clip_analysis_frame, self.frame_analysis_frame, self.bands_analysis_frame]
        for frame in frames:
            for _, child in frame.children.items():
                if isinstance(child, tk.Button):
                    child.configure(state=tk.NORMAL)
    
    def draw_waveform(self):
        ax = self.fig.gca()
        ax.clear()

        data, time = self.analyzer.waveform()

        ax.plot(time, data, label="Przebieg sygnału")

        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid()
        # ax.set_title("Przebieg czasowy sygnału audio")

        self.canvas.draw()

        plt.close(self.fig)


    def show_parameters(self):
        params = self.analyzer.calculate_parameters()
        
        self.params_window = tk.Toplevel(self.root, padx=40, pady=10)
        self.params_window.title("Parametry klipu")
        self.params_window.geometry(f"{Config.MENU_WIDTH}x{Config.MENU_HEIGHT}")
        self.root.eval(f'tk::PlaceWindow {str(self.params_window)} center')

        self.params_label_frame = tk.Frame(self.params_window, width=Config.MENU_WIDTH / 2)
        self.params_label_frame.pack(side=tk.LEFT, anchor='ne')

        self.params_value_frame = tk.Frame(self.params_window, width=Config.MENU_WIDTH / 2)
        self.params_value_frame.pack(side=tk.RIGHT, anchor='nw', padx=(0, 40))

        for key, value in params.items():
            label = tk.Label(self.params_label_frame, text=f"{key}:")
            label.pack(pady=5, anchor='e')

            if not isinstance(value, str):
                value = round(float(value), 3)
            param = tk.Label(self.params_value_frame, text=f"{value}")
            param.pack(pady=5, anchor='w')

        self.params_window.transient(self.root)
        self.params_window.grab_set()
    
    def window_analysis_window(self):
        FrameAnalysisWindow(self.root, self.analyzer)

    def spectrogram_window(self):
        SpectogramAnalysisWindow(self.root, self.analyzer)

    def show_settings(self):
        self.config_window = tk.Toplevel(self.root, padx=40, pady=10)
        self.config_window.title("Ustawienia parametrów")
        self.config_window.geometry(f"{Config.MENU_WIDTH}x{Config.MENU_HEIGHT}")
        self.root.eval(f'tk::PlaceWindow {str(self.config_window)} center')

        self.frame_size_var = tk.StringVar(value=str(int(Config.FRAME_SIZE * 1000)))
        self.min_f0_var = tk.StringVar(value=str(Config.MIN_F0))
        self.max_f0_var = tk.StringVar(value=str(Config.MAX_F0))
        self.entropy_k_var = tk.StringVar(value=str(Config.ENTROPY_K))

        tk.Label(self.config_window, text="Rozmiar ramki (ms):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.frame_size_var).pack()

        tk.Label(self.config_window, text="Min. F0 (hz):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.min_f0_var).pack()

        tk.Label(self.config_window, text="Maks. F0 (hz):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.max_f0_var).pack()

        tk.Label(self.config_window, text="Rozmiar segmentu K (w entropii energii):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.entropy_k_var).pack()

        tk.Button(self.config_window, text="Zapisz", command=self.save_config).pack(pady=15)
        self.config_window.bind('<Return>', lambda event: self.save_config())

    def save_config(self):
        try:
            Config.FRAME_SIZE = int(self.frame_size_var.get()) / 1000
            Config.MIN_F0 = int(self.min_f0_var.get())
            Config.MAX_F0 = int(self.max_f0_var.get())
            Config.ENTROPY_K = int(self.entropy_k_var.get())

            if Config.MIN_F0 == 0:
                Config.MIN_F0 += 1
            if Config.MAX_F0 == 0:
                Config.MAX_F0 += 1
            assert Config.MAX_F0 >= Config.MIN_F0

            if self.filepath is not None:
                try:
                    self.analyzer = FileAnalyzer(self.filepath)
                except Exception as e:
                    print(e)
            
            self.config_window.destroy()
        except (ValueError, AssertionError) as e:
            print(e)
            tk.messagebox.showerror("Błąd", "Wprowadzono niepoprawne wartości liczbowe")

    def plot_waveform_freq(self):
        energy, time = self.analyzer.waveform_freq()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, energy)
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Amplituda")
        plt.title("Amplituda sygnału na poziomie klipu")
        plt.xscale('log')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_volume(self):
        volume, time = self.analyzer.volume()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, volume, label="Głośność na poziomie ramek")
        plt.xlabel("Czas [s]")
        plt.ylabel("Głośność")
        plt.title("Głośność na poziomie ramek")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_fc(self):
        fc, time = self.analyzer.fc()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, fc)
        plt.xlabel("Czas [s]")
        plt.ylabel("Hz")
        plt.title("Frequency centroid na poziomie ramek")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_bandwidth(self):
        bw, time = self.analyzer.bw()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, bw)
        plt.xlabel("Czas [s]")
        plt.ylabel("Hz")
        plt.title("Efektywne szerokość pasma na poziomie ramek")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_ersb(self):
        ersb1, ersb2, ersb3, time = self.analyzer.ersb()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, ersb1, label="ERSB1")
        plt.plot(time, ersb2, label="ERSB2")
        plt.plot(time, ersb3, label="ERSB3")
        plt.xlabel("Czas [s]")
        plt.ylabel("Stosunek energii")
        plt.title("Stosunek energii w pasmach częstotliwości (ERSB) w czasie")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_sfm(self):
        sfm, time = self.analyzer.sfm()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, sfm)
        plt.xlabel("Czas [s]")
        plt.ylabel("SFM")
        plt.title("Miara płaskości widma w czasie")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_sfm_bands(self):
        sfmb1, sfmb2, sfmb3, time = self.analyzer.sfm_bands()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, sfmb1, label="SFM B1")
        plt.plot(time, sfmb2, label="SFM B2")
        plt.plot(time, sfmb3, label="SFM B3")
        plt.xlabel("Czas [s]")
        plt.ylabel("SFM")
        plt.title("Miara płaskości widma w pasmach częstotliwości (SFM subbands)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_scf_bands(self):
        scfb1, scfb2, scfb3, time = self.analyzer.scf_bands()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, scfb1, label="SCF B1")
        plt.plot(time, scfb2, label="SCF B2")
        plt.plot(time, scfb3, label="SCF B3")
        plt.xlabel("Czas [s]")
        plt.ylabel("SCF")
        plt.title("Spectral Crest Factor w pasmach częstotliwości w czasie")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_f0(self):
        f0, time = self.analyzer.f0()

        plt.figure(figsize=(10, 4))
        plt.plot(time, f0)
        plt.xlabel("Czas [s]")
        plt.ylabel("Częstotliwość [Hz]")
        plt.title("Częstotliwość krtaniowa f0 na podstawie cepstrum")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # def plot_voicing(self):
        
    #     labels, time = self.analyzer.voicing()
    #     volume, _ = self.analyzer.volume()

    #     color_map = {
    #         'voiced': 'green',
    #         'unvoiced': 'red',
    #         'ambiguous': 'gray'
    #     }

    #     plt.figure(figsize=(10, 4))

    #     # to create the table
    #     for label, color in color_map.items():
    #         plt.axvspan(0, 0, color=color, alpha=0.4, label=label)

    #     # (un)voiced segments
    #     delta = time[1] - time[0]
    #     for t, label in zip(time, labels):
    #         if label == 'ambiguous':
    #             continue
    #         t_start = t
    #         t_end = t_start + delta
    #         plt.axvspan(t_start, t_end, color=color_map.get(label), alpha=0.4)

    #     plt.plot(time, volume, label="volume", alpha=0.8)

    #     plt.xlabel('Czas [s]')
    #     plt.ylabel('Głośność')
    #     plt.title('Podział na fragmenty dźwięczne i bezdźwięczne')
    #     plt.xlim(time[0], time[-1] + delta)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    def save_file(self, clip=False):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save file as"
        )

        if file_path:
            dataframe = self.prepare_params(clip=clip)
            dataframe.to_csv(file_path)

    def prepare_params(self, clip=False):
        if clip:
            params = self.analyzer.calculate_parameters()
            return pd.DataFrame(params, index=[0])
        else:
            params = {}
            ste, params['time'] = self.analyzer.ste()
            params['ste'] = ste
            params['zcr'], _ = self.analyzer.zcr()
            params['volume'], _ = self.analyzer.volume()
            silence, _ = self.analyzer.silence()
            params['silence'] = silence.astype(int)
            params['voicing'], _ = self.analyzer.voicing()
            params['f0'], _ = self.analyzer.f0(method='cor')
            return pd.DataFrame(params)


class FrameAnalysisWindow:
    def __init__(self, master, analyzer: FileAnalyzer):
        self.analyzer = analyzer
        self.frame_index = 0
        self.window_type = 'boxcar'

        self.window = tk.Toplevel(master)
        self.window.title("Analiza wybranej ramki")

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.first_update = True
        self.y_lim = (0,0)
        self.x_lim = (0,0)

        self.create_widgets()
        self.update_plots()

    def on_close(self):
        plt.close(self.fig)
        self.window.destroy()

    def create_widgets(self):
        window_label = tk.Label(self.window, text="Funkcja okna:")
        window_label.pack()
        
        self.window_options = ['boxcar', 'triang', 'hamming', 'hann', 'blackman', 'bartlett',
                                'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
                                'barthann', 'cosine', 'exponential', 'tukey', 'taylor', 'lanczos']
        self.window_dropdown = ttk.Combobox(self.window, values=self.window_options)
        self.window_dropdown.set(self.window_type)
        self.window_dropdown.bind("<<ComboboxSelected>>", self.on_window_change)
        self.window_dropdown.pack()

        # self.slider = tk.Scale(self.window, from_=0, to=len(self.analyzer.frames()) - 1,
        #                        orient=tk.HORIZONTAL, label="Numer ramki", command=self.on_slider_change)
        # self.slider.pack(fill='x', padx=10, pady=5)
                # Start and End frame sliders
        self.start_slider = tk.Scale(self.window, from_=0, to=len(self.analyzer.frames()) - 2,
                                     orient=tk.HORIZONTAL, label="Start ramki", command=self.on_slider_change)
        self.start_slider.pack(fill='x', padx=10)

        self.end_slider = tk.Scale(self.window, from_=1, to=len(self.analyzer.frames()) - 1,
                                   orient=tk.HORIZONTAL, label="Koniec ramki", command=self.on_slider_change)
        self.end_slider.set(len(self.analyzer.frames()) - 1)
        self.end_slider.pack(fill='x', padx=10)

        self.selected_range = (0, len(self.analyzer.frames())-1)

        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

    def on_window_change(self, event):
        self.window_type = self.window_dropdown.get()
        self.update_plots()

    def on_slider_change(self, value):
        start = self.start_slider.get()
        end = self.end_slider.get()
        if end <= start:
            if self.last_end != end:
                start = end - 1
                self.start_slider.set(start)
            else:
                end = start + 1
                self.end_slider.set(end)
        self.frame_index = start  # could still use first frame in range
        self.selected_range = (start, end)
        self.last_end = end
        self.update_plots()

    def update_plots(self):
        start, end = self.selected_range
        frames = self.analyzer.frames()[start:end]
        frame = np.array(frames).flatten()

        windowed_frame = self.analyzer.apply_window(frame, self.window_type)
        fft_result, freqs = self.analyzer.waveform_freq(windowed_frame)
        # freqs = freqs[1:]  # to avoid frequency 0 in log scale
        # fft_result = fft_result[1:]

        if self.first_update:
            self.x_lim = (min(freqs) + 0.1, max(freqs))
            self.first_update = False
        
        # FFT
        self.ax_time.clear()
        self.ax_freq.clear()

        time = self.analyzer.time(len(self.analyzer.frames()))[start:end]
        time_axis = np.linspace(time[0], time[-1] + Config.FRAME_SIZE, len(frame))

        self.ax_time.plot(time_axis, windowed_frame)
        self.ax_time.set_title("Sygnał w dziedzinie czasu")
        self.ax_time.set_xlabel("Czas [s]")
        self.ax_time.set_ylabel("Amplituda")

        self.ax_freq.plot(freqs, fft_result)
        self.ax_freq.set_title("Widmo częstotliwościowe (FFT)")
        self.ax_freq.set_xlabel("Częstotliwość [Hz]")
        self.ax_freq.set_ylabel("Amplituda")
        self.ax_freq.set_xscale('log')
        self.ax_freq.set_xlim(self.x_lim)

        self.fig.tight_layout()
        self.canvas.draw()


class SpectogramAnalysisWindow:
    def __init__(self, master, analyzer: FileAnalyzer):
        self.analyzer = analyzer
        self.window_type = 'boxcar'
        self.overlap_value = 0.5

        self.window = tk.Toplevel(master)
        self.window.title("Spektogram")

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.first_update = True
        self.y_lim = (0,0)
        self.x_lim = (0,0)

        self.create_widgets()
        self.update_plots()

    def on_close(self):
        plt.close(self.fig)
        self.window.destroy()

    def create_widgets(self):
        # Dropdown do wyboru okna
        window_label = tk.Label(self.window, text="Funkcja okna:")
        window_label.pack()
        
        self.window_options = ['boxcar', 'triang', 'hamming', 'hann', 'blackman', 'bartlett',
                                'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
                                'barthann', 'cosine', 'exponential', 'tukey', 'taylor', 'lanczos']
        self.window_dropdown = ttk.Combobox(self.window, values=self.window_options)
        self.window_dropdown.set(self.window_type)
        self.window_dropdown.bind("<<ComboboxSelected>>", self.on_window_change)
        self.window_dropdown.pack()

        self.overlap_label = tk.Label(self.window, text="Overlap (%)")
        self.overlap_label.pack()
        self.overlap_slider = tk.Scale(self.window, from_=0, to=90, orient=tk.HORIZONTAL, command=self.on_overlap_change)
        self.overlap_slider.set(int(self.overlap_value * 100))
        self.overlap_slider.pack()

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.colorbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

    def on_window_change(self, event):
        # get overlap value 
        self.window_type = self.window_dropdown.get()
        self.update_plots()

    def on_overlap_change(self, val):
        self.overlap_value = self.overlap_slider.get() / 100
        self.update_plots()

    def update_plots(self):
        spec_db, time_axis, freq_axis = self.analyzer.spectrogram(self.window_type, self.overlap_value)

        self.ax.clear()
        # if self.colorbar:
        #     self.colorbar.ax.remove()

        img = self.ax.imshow(spec_db, origin='lower', aspect='auto',
                             extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
        self.ax.set_xlabel("Czas [s]")
        self.ax.set_ylabel("Częstotliwość [Hz]")
        self.ax.set_title("Spektrogram")

        if self.first_update:
            self.colorbar = self.fig.colorbar(img, ax=self.ax, label="Amplituda [dB]")
            self.first_update = False

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    app.root.mainloop()
