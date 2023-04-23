import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import scipy.signal as signal

def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """

    # corr = np.correlate(x, y, mode='full')
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr


def point_add(data, fname):
    analy_info = {}
    xdata, ydata = data['Time'], data['Ampl']

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

                plt.savefig(f'{fname.split(".")[0]}.png')
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

    if len(point_x)>1:

        point_x.sort()
        datas = int(((point_x[0] - xdata[0])/analy_info['dt']))
        datae = int(((point_x[-1] - xdata[0])/analy_info['dt']) + 1)

        new_y = data.iloc[datas:datae, 1]
        y = new_y - np.mean(new_y)

        lag, corr = xcorr(y, y)

        ttt = lag * analy_info['dt']

        c = np.abs(corr)

        peakind = signal.find_peaks(c, height=100, distance=8000)

        plt.plot(ttt, c)
        plt.plot(ttt[peakind[0]], peakind[1]['peak_heights'], 'ro', fillstyle='none')
        plt.title('Rxx auto-correlation result')
        plt.xlabel('lags')
        plt.ylabel('auto-correlation')
        plt.grid()
        plt.savefig(f'{fname.split(".")[0]}_auto_corr.png')
        plt.show()

    else:
        return



def main():
    file_list = glob.glob(os.path.join('*.csv'))

    for f in file_list:
        tmp = pd.read_csv(f, skiprows=4, header=0)
        point_add(data=tmp,fname=f)




if __name__ == '__main__':
    main()