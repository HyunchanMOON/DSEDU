import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import scipy.signal as signal
import tkinter as tk
import tkinter.filedialog


def fft_cal(data, datas, datae, window_sel=1, select_freq=0, f_req=5): # fft 를 진행하는 함수
    window_dict = {0: 'no window', 1: 'hanning', 2: 'tukey'}
    figsave = 1


    data = pd.read_csv(data, header=None, encoding='utf-8', sep=' ')
    data = data.values.astype(np.float64)
    # data = np.loadtxt(data) # dat 파일을 load 한다.

    xdata, ydata = data[:, 0], data[:, 1] # 첫번째 항목이 시간, 두번째 항목이 y 값이다.
    dt = xdata[1] - xdata[0] # dt 계산

    new_y = data[datas:datae, 1] # 시작, 끝 점으로 ydata parsing

    y = new_y - np.mean(new_y) # fft 계산을 위해 mean 값을 빼준다.

    # window 선택 인자에 따라 설정해준다.
    if window_sel == 1:
        w = np.hanning(len(y)) # hanning window
        y = y * w
    elif window_sel == 2:
        w = signal.windows.tukey(len(y)) # tukey window
        y = y * w

    sampletime = dt
    samplenumber = len(y)

    t = xdata[datas:datae]

    samplerate = np.round(1 / (samplenumber * sampletime * 100),3) # sample rate 걔산

    f = np.arange(0, samplerate * 100 * samplenumber, samplerate)

    ### FFT 시작
    fy = np.abs(np.fft.fft(y, len(y) * 100))

    ### 윈도우에 따른 신호 크기 보상

    if window_sel == 0:
        fy = (fy /samplenumber) * 2  # 일반

    elif window_sel == 1:
        fy = (fy /samplenumber) * 4  # 해닝윈도우

    elif window_sel == 2:
        fy = (fy /samplenumber) * (8 / 3)  # 터키윈도우

    fy = np.round(fy,6)


    if select_freq == 0:  # Automatic search
        fy[0] = 0
        MaxAmp1 = max(fy)
        MaxFreq1 = np.where(fy==MaxAmp1)[0][0]

        a = MaxFreq1
        b = 2*MaxFreq1 - 1

    elif select_freq == 1:  # Forced search
        a = round((f_req * 1000000) / samplerate) + 1
        b = round(2 * (f_req * 1000000) / samplerate) + 1

    A1 = fy[a]
    A1_square = np.round(np.square(fy[a]),5)
    A2 = fy[b]

    if len(f) != len(fy):
        f = f[:-1]


    fig, ax = plt.subplots(3, 1, figsize=(10,8), constrained_layout=True)

    ax[0].plot(t, y)
    ax[0].set_title(f'Oscilloscope Data')
    ax[0].set_xlabel(f'Time')
    ax[0].set_ylabel(f'Amplitude')
    ax[0].grid()

    ax[1].plot(f / 1000000, fy)
    ax[1].set_xlim([0, 15])
    ax[1].set_ylim([0, max(fy) * 1.2])
    ax[1].set_title(f'Result of FFT')
    ax[1].set_xlabel(f'Frequency(MHz)')
    ax[1].set_ylabel(f'Magnitude')
    ax[1].grid()

    # ax[2].set_ylim([0, max(np.log(fy)) * 1.2])

    ax[2].plot(f / 1000000, fy)
    ax[2].set_xlim([0, 15])
    ax[2].set_yscale('log')
    ax[2].set_title(f'Result of FFT Logscale')
    ax[2].set_xlabel(f'Frequency(MHz)')
    ax[2].set_ylabel(f'Magnitude')
    ax[2].grid()

    plt.savefig(f'{window_dict[window_sel]}_and_fft.png')

    plt.show(block=False)
    plt.pause(1.5)
    plt.close()

    return A1, A1_square, A2, f[a], f[b]


def point_add(data):

    analy_info = {}
    xdata, ydata = data[:, 0], data[:, 1]

    analy_info['dt'] = xdata[1] - xdata[0]

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid(True)
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.title('Oscilloscope signal')
    ax.set_aspect('auto', adjustable='box')
    line, = ax.plot(xdata, ydata)

    point_x, point_y = [], []

    def add_point(event):
        if event.inaxes != ax:
            return

        # button 1: 마우스 좌클릭
        if event.button == 1:
            x = event.xdata
            # y = event.ydata
            if len(point_x)>1:
                plt.close()
                return
            else:
                point_x.append(x)
                point_y.append(0)
                ax.plot(point_x, point_y, 'r*')
                plt.draw()

        # button 3: 마우스 우클릭 시 기존 입력값 삭제
        if event.button == 3:
            if len(point_x) >=1:
                point_x.pop()
                point_y.pop()
            ax.clear()
            line, = ax.plot(xdata, ydata)
            plt.grid(True)
            plt.xlabel('Time (sec)')
            plt.ylabel('Amplitude')
            plt.title('Oscilloscope signal')
            ax.plot(point_x, point_y, 'r*')
            plt.draw()

        # 마우스 중간버튼 클릭 시 종료하기
        if event.button == 2:
            plt.disconnect(cid)
            plt.close()

    cid = plt.connect('button_press_event', add_point)
    plt.show()
    plt.close()

    if len(point_x)>1: # 2 점이 올바르게 선택 되었을 경우에만 시작점 끝점 계산함.
        point_x.sort()
        datas = int(((point_x[0] - xdata[0])/analy_info['dt']))
        datae = int(((point_x[-1] - xdata[0])/analy_info['dt']))
        return datas, datae
    else:
        return None, None



def main():

    root = tk.Tk()
    root.geometry("0x0+100+100") # file open dialog 사이즈
    filename = tk.filedialog.askopenfilename(title="open file", # file open dialog 생성하고 open 한다. .dat 파일만 보이도록 함
                                             filetypes=(("dat file", "*.dat"), ("all files", "*.*")))
    root.destroy()

    if filename != '': # 파일이 선택 됐을 때만 실행

        # dat_f = open(filename)
        # dat_arr = np.array([])
        #
        # for i, data in enumerate(dat_f):
        #     d = np.array(data.split())
        #     d = d.astype(np.float64)
        #     d = d.reshape(1,-1)
        #     dat_arr = np.append(dat_arr, d)
        #
        # print(dat_arr)
        data = pd.read_csv(filename, header=None, encoding='utf-8', sep=' ')

        data = data.values.astype(np.float64)

        # data = np.loadtxt(filename) # np library를 이용해서 data file 한개만 load 한다.

        datas, datae = point_add(data=data) # 특정 2 지점을 선택해서 시작점 끝점을 추출한다.

        data_list = glob.glob(os.path.join('./', '*.dat')) # 경로에 dat 파일 모두 list에 담는다.

        result_df = pd.DataFrame(columns=['data name', 'F.Freq', 'S.Freq', 'A1', 'A1^2', 'A2'], index=list(range(len(data_list)))) # 결과를 저장하기 위한 data frame 생성
        A1_list, A1_square_list, A2_list = [], [], [] # A1, A1^2, A2 저장하기 위한 list 생성

        if datas is not None:
            for i, d in enumerate(data_list): # 각각의 dat 파일 순차적으로 모두 불러온다.
                fname = d.split('\\')[-1] # file name 만 추출
                A1, A1_square, A2, F, S = fft_cal(data=d, datas=datas, datae=datae, window_sel=1, select_freq=0, f_req=5) # 해당 dat 파일의 정보와 위에서 설정한 시작/끝 점 , fft 를 위한 parameter를 설정한다.
                result_df.iloc[i, :] = fname, np.round(F/1000000,2), np.round(S/1000000,2), A1, A1_square, A2 # 각각의 결과 순차적으로 결과를 저장하는 data frame에 채워준다.
                A1_list.append(A1) # 단일 A1 값을 A1_list에 추가한다.
                A1_square_list.append(A1_square) # 단일 A1_square 값을 A1_square_list에 추가한다.
                A2_list.append(A2) # 단일 A2 값을 A2_list에 추가한다.

            print(result_df) # 최종 결과 출력

            plt.figure()
            plt.grid(True)
            plt.rc('font')
            plt.box(True)
            plt.gca().set_facecolor((1.0, 1.0, 1.0))
            plt.scatter(A1_square_list, A2_list, marker='*', color='b')

            A1_square_list = np.array(A1_square_list)
            A1_list = np.array(A1_list)
            A2_list = np.array(A2_list)

            One = np.ones((len(data_list), 1)) # 1로 구성된 data 개수 만큼의 배열 생성
            Re_A1_square = np.concatenate((One, A1_square_list.reshape(-1, 1)), axis=1) # 열을 기준으로 concat
            Re_A2 = A2_list.reshape(-1, 1) # 행렬 계산을 위해 차원을 늘려줌

            Coef = np.linalg.pinv(Re_A1_square.T @ Re_A1_square) @ Re_A1_square.T @ Re_A2 # numpy 행렬 연산(@)

            AproxX = np.arange(A1_square_list[0], A1_square_list[len(data_list) - 1], 0.1)
            ApproY = Coef[1] * AproxX + Coef[0]

            print('y =', Coef[1], 'x +', Coef[0])

            plt.plot(AproxX, ApproY, 'r')
            plt.title('Least squares')
            plt.xlabel('A1^2')
            plt.ylabel('A2')
            plt.xlim([0, max(A1_square_list) * 1.2])
            plt.ylim([0, max(A2_list) * 1.2])
            plt.grid(True)

            plt.show()

if __name__ == '__main__':
    main()


