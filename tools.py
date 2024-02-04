import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.app_logo import add_logo

# function
# page config
def page_config(title):
    st.set_page_config(page_title=title, page_icon="⚠️")
    hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
	        header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True) 
    # logo
    add_logo("image/code.png", height=100) 
    # sidebar
    with st.sidebar:
        st.info("⬆⬆ Pick a menu above! ⬆⬆")

# banjir
def klasifikasi_banjir(X):
    X['status_pred'] = np.where(X['height'] < 90, 4, 
                        np.where(X['height'] < 130, 3, 
                                 np.where(X['height'] < 160, 2,
                                          1)))
    df_klasifikasi = X
    return df_klasifikasi

def get_X_prediksi(data, date):
    data_history = data.loc[data['datetime'] <= date].head(260).sort_values(by=['datetime']).reset_index(drop=True) # 144+108=252, 260>252
    data_history['height_diff_18h (cm)'] = data_history['height (cm)'] - data_history['height (cm)'].shift(108) # ekstraksi fitur
    data_history = data_history.dropna().tail(144).reset_index(drop = True)
    data_history = data_history[['datetime',
                                 'height (cm)',
                                 'precip (mm)',
                                 'visibility (km)',
                                 'windgust (kph)',
                                 'height_diff_18h (cm)']]
    return data_history

def prediksi_banjir(data, date, X, scaler_X, scaler_y, model):
    # X
    X_prediksi = X.drop(columns=['datetime']).rename(columns={
                                    'height (cm)':'height',
                                    'precip (mm)' : 'precip',
                                    'visibility (km)' : 'visibility',
                                    'windgust (kph)':'windgust',
                                    'height_diff_18h (cm)':'height_diff_18h'})
    # scaling
    X_prediksi_scaled = scaler_X.transform(X_prediksi)
    # reshape 
    X_prediksi_scaled = X_prediksi_scaled.reshape(1,144,5)
    # predict
    y_prediksi = model.predict(X_prediksi_scaled,verbose=0)
    # reshape
    y_prediksi = y_prediksi.reshape(36,1)
    # inverse scaling
    y_prediksi_inverse = scaler_y.inverse_transform(y_prediksi).round(2)
    y_prediksi_inverse = pd.DataFrame(y_prediksi_inverse, columns = ['height']) # y_pred
    # DATA FUTURE
    data_future = data.loc[data['datetime'] > date].tail(36).sort_values(by=['datetime']).reset_index(drop=True)# 36=step
    data_future = data_future[['datetime','height (cm)']].rename(columns = {'height (cm)':'height_true (cm)'})
    # DF PRED
    if data_future.shape[0] == 36:
        df_pred = data_future.join(y_prediksi_inverse)
    else:
        df_pred = y_prediksi_inverse
    return df_pred

def get_info_banjir2(y_klasifikasi, y_pred_status): # informasi prediksi tab 2
    date = y_klasifikasi['date'][0]
    height = y_klasifikasi['height'][0]
    status = y_klasifikasi['status_pred'][0]
    
    st.write("**Prediction info 💬**")
    col_title, col_text = st.columns([1,8])

    col_title.write('Datetime')
    col_text.write(f': {str(date)} WIB') 

    col_title.write('Location')
    col_text.write(': Padang, Indonesia')

    col_title.write('Height')
    col_text.write(f': {str(height.round(2))} cm')

    # status klasifikasi
    siaga4 = ':green[SIAGA 4]'
    siaga3 = ':yellow[SIAGA 3]'
    siaga2 = ':orange[SIAGA 2]'
    siaga1 = ':red[SIAGA 1]'
    normal = ':green[NORMAL]'
    waspada = ':yellow[WASPADA]'
    siaga = ':orange[SIAGA]'
    bahaya = ':red[BAHAYA]'

    col_title.write('Status')
    col_text.write(f': SIAGA {status}')

    y_pred_max = y_pred_status['height_pred (cm)'].max()
    status_pred_max = y_pred_status['status_pred'].min()

    if status_pred_max == 4:
        col_title.write("Message")
        col_text.write(f': {[normal]} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga4})')
    elif status_pred_max == 3:
        col_title.write("Message")
        col_text.write(f': {[waspada]} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga3})')
    elif status_pred_max == 2:
        col_title.write("Message")
        col_text.write(f': {[siaga]} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga2})')
    elif status_pred_max == 1:
        col_title.write("Message")
        col_text.write(f': {[bahaya]} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga1})')
             

def get_info_banjir3(y_klasifikasi, y_pred_status):
    date = y_klasifikasi['date'][0] # info datetime
    height = y_klasifikasi['height'][0] # info height
    status = y_klasifikasi['status_pred'][0]
    status_siaga = f"SIAGA {status}"

    # status klasifikasi
    siaga1 = 'SIAGA 1'
    siaga2 = 'SIAGA 2'
    siaga3 = 'SIAGA 3'
    siaga4 = 'SIAGA 4'   
    normal = "*[NORMAL]*"
    waspada = "*[WASPADA]*"
    siaga = "*[SIAGA]*"
    bahaya = "*[BAHAYA]*"

    y_pred_max = (y_pred_status['height_pred (cm)'].max()).round(2)
    status_pred_max = y_pred_status['status_pred'].min()

    if status_pred_max == 4:
        msg_line1 = f"{normal} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga4})"
    elif status_pred_max == 3:
        msg_line1 = f"{waspada} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga3})"
    elif status_pred_max == 2:
        msg_line1 = f"{siaga} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga2})"
    elif status_pred_max == 1:
        msg_line1 = f"{bahaya} Diprediksi ketinggian air maksimal dalam 6 jam ke depan adalah {y_pred_max:.2f} cm ({siaga1})"
    
    return date, height, status_siaga, msg_line1


# gempa
def klasifikasi_gempa(X, scaler_X, model):
    X_data=X.iloc[:,:6]
    X_scaled = scaler_X.transform(X_data)
    X_scaled=X_scaled.reshape(1,1,6)
    # predict
    y_pred = model.predict(X_scaled, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    # df
    df_y_pred = pd.DataFrame(y_pred, columns=['result_pred'])
    df_klasifikasi = X.join(df_y_pred)
    return df_klasifikasi
def get_info_gempa(data):
    data = data.rename(columns = {'aX':'aX (g)',
                                  'aY':'aY (g)',
                                  'aZ':'aZ (g)',
                                  'gX':'gX (deg/s)',
                                  'gY':'gY (deg/s)',
                                  'gZ':'gZ (deg/s)'})
    status_pred = data['result_pred'][0]
    if status_pred == 1:
        st.error("Result: Earthquake")
    else:
        st.success("Result: Non-Earthquake")
    with st.expander("Explore classified data"):
                col_ex1,col_ex2 = st.columns([3,1])
                col_ex1.dataframe(data)
                col_ex2.markdown(":green[0 = Non-Earthquake]")
                col_ex2.markdown(":red[1 = Earthquake]")
