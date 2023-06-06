import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 음수와 한글 표기 가능하도록 setup
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus']= False

pd.set_option('display.max_columns', 50)

def main(df_in):

    new_cols = ['code', 'n_house', 'type', 'region', 'region_park', 'income', 'population', 'middle_age',
                'area_exclusive', 'vacancy', 'deposit', 'rent', 'subway_station', 'bus_station', 'car',
                'parking_lot']  # 열이름을 변경하기 위한 list

    df_in.columns = new_cols  # 열이름 변경

    print(df_in.info()) #모든 column type 확인
    print(df_in.isna().sum()) # 결측치 확인

    print("######### drop na in bus_station #########")
    df_in.dropna(subset=['bus_station'], inplace=True) # 결측치 1개 이므로 그냥 drop 한다.
    print(df_in.isna().sum())  # 재확인
    # df_in['deposit'] = pd.to_numeric(df_in['deposit']) #  object type을 갖고 있는 열 중 하나인 deposit을 numeric type으로 변경하려 했으나 '-' 값이 발견되어 오류 발생

    print(df_in[df_in['deposit'] == '-'])  # - 값을 갖고 있는 부분 확인

    df_in['region_park'] = df_in['region_park'].str.replace(',', '')  # region_park에 대해 ',' -> '' replace 진행
    df_in['income'] = df_in['income'].str.replace(',', '')  # income에 대해 ',' -> '' replace 진행
    df_in['population'] = df_in['population'].str.replace(',', '')  # population에 대해 ',' -> '' replace 진행

    df_in['deposit'] = df_in['deposit'].str.replace('-', '0')  # deposit에 대해 ',' -> '' replace 진행
    df_in['rent'] = df_in['population'].str.replace('-', '0')  # rent에 대해 ',' -> '' replace 진행

    for c in df_in.columns:
        df_in[c] = pd.to_numeric(df_in[c], errors='ignore')  # 모든 column에 대해 숫자로 변경이 가능한 column은 정수형으로 변경한다. 불가능한경우 원상태를 유지하도록한다.

    print(df_in.info()) # 변경후 재확인
    print(df_in.describe()) # 수치형으로 변환된 column 들의 기초 통계 summary

    plt.figure(figsize=(20, 8))

    g1 = sns.heatmap(df_in.corr(), annot=True) # 모든 값간의 correlation을 구하고 값을 heat map 으로 보여준다.
    plt.title("All Feature Correlation Map")


    valid_cols = ['area_exclusive', 'vacancy', 'subway_station', 'bus_station', 'parking_lot']
    df_parse = df_in[valid_cols]
    print(df_parse.head())

    # 각 독립변수의 특성 별 parking lot 을 확인 하려면 각 독립변수로 groupby 후 해당 독립변수의 값에 따른 parkinglot 을 summation 해야한다.
    area_exclusive_grp = df_parse.groupby(['area_exclusive']).sum() # area_exclusive 값에 대한 groupby 후 나머지 columns에 대한 합을 구한다.
    vacancy_grp = df_parse.groupby(['vacancy']).sum() # vacancy 값에 대한 groupby 후 나머지 columns에 대한 합을 구한다.
    subway_station_grp = df_parse.groupby(['subway_station']).sum() # subway_station 값에 대한 groupby 후 나머지 columns에 대한 합을 구한다.
    bus_station_grp = df_parse.groupby(['bus_station']).sum() # bus_station 값에 대한 groupby 후 나머지 columns에 대한 합을 구한다.

    figure, ax = plt.subplots(nrows=4, ncols=1) # figure 생성 및 각 독립변수 4개에 대해 모두 보여주기위한 subplot을 생성한다.
    figure.tight_layout() # 각 subplot 간 침범하지 않도록 설정

    ## groupby 된 각 독립 변수와 parking lot 간의 bar plot을 그리고 색을 각 다르게 설정한다. ##
    area_exclusive_grp.plot(kind='bar', y='parking_lot', ax=ax[0],color='b')
    vacancy_grp.plot(kind='bar', y='parking_lot', ax=ax[1],color='r')
    subway_station_grp.plot(kind='bar', y='parking_lot', ax=ax[2],color='g')
    bus_station_grp.plot(kind='bar', y='parking_lot', ax=ax[3],color='k')
    plt.suptitle('parking lot according to each characteristic')

    figure2, ax1 = plt.subplots(nrows=4, ncols=1) # figure 생성 및 각 독립변수 4개에 대해 모두 보여주기위한 subplot을 생성한다.
    figure2.tight_layout() # 각 subplot 간 침범하지 않도록 설정

    ## groupby 된  각 독립 변수와 parking lot 간의 line plot을 그리고 색을 각 다르게 설정한다. ##
    area_exclusive_grp.plot( y='parking_lot', ax=ax1[0],color='b')
    vacancy_grp.plot( y='parking_lot', ax=ax1[1],color='r')
    subway_station_grp.plot(y='parking_lot', ax=ax1[2],color='g')
    bus_station_grp.plot( y='parking_lot', ax=ax1[3],color='k')
    plt.suptitle('parking lot according to each characteristic')

    ## groupby 하지않은 모든 값이 각 독립 변수와 parking lot 간의 bar plot을 그리고 색을 각 다르게 설정한다. ##
    figure3, ax2 = plt.subplots(nrows=4, ncols=1)
    figure3.tight_layout()
    df_parse.plot.scatter(x='area_exclusive', y='parking_lot',ax=ax2[0],color='b', s=20)
    df_parse.plot.scatter(x='vacancy', y='parking_lot',ax=ax2[1],color='r', s=20)
    df_parse.plot.scatter(x='subway_station', y='parking_lot',ax=ax2[2],color='g', s=20)
    df_parse.plot.scatter(x='bus_station', y='parking_lot',ax=ax2[3],color='k', s=20)
    plt.suptitle('Scatter plot : parking lot according to each characteristic')

    ## 각 변수간 scatter 및 regression plot을 통해 상관성을 확인한다. ##
    g2 = sns.pairplot(data=df_parse, kind='reg')
    g2.fig.suptitle("Pair plot : parking lot according to each characteristic")  # y= some height>1

    ## 각 변수들의 histogram을 plot 한다. ##
    df_parse.hist(density= True)


    ## subway station 수는 0~3의 unique한 값들로 구성되어있는데 각 값이 차지하는 비율이 어떻게 되는지 pie 차트로 보여주기 위한 과정##
    # unique 한 값들 / 모든 값
    subway_ratio = df_parse['subway_station'].value_counts()/df_parse['subway_station'].value_counts().sum()
    # print(subway_ratio)

    labels = ['Station_num 0', 'Station_num 1', 'Station_num 2', 'Station_num 3']
    plt.figure()
    plt.pie(subway_ratio, labels=labels, autopct='%.1f%%')
    plt.title('Percentage of the number of subway stations')
    plt.show()


if __name__ == "__main__":

    parking_df = pd.read_csv("parking_data (1).csv", encoding='cp949')  # 한글 읽기 위한 encoding type 설정
    print(parking_df.head())
    main(df_in = parking_df)