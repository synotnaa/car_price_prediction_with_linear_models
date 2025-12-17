import re
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# Конфигурация страницы Streamlit
st.set_page_config(
    page_title='Car price prediction',
    page_icon='🚗',
    layout='wide',
)


# Пути к модели и данным для предобработки
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR/'models'/'linear_model.pkl'
TRAIN_URL = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
TARGET_COL = 'selling_price'


# Предобработка по аналогии с ноутбуком
def magic_parsing(cell):
    if pd.isna(cell):
        return pd.Series([np.nan, np.nan])

    cell_val = str(cell).lower().replace(' ', '').replace(',', '')
    moment_search = re.search(r'([\d\.]+)', cell_val)

    if not moment_search:
        return pd.Series([np.nan, np.nan])

    torque = float(moment_search.group(1))
    if 'kgm' in cell_val:
        torque *= 9.81

    rpm = np.nan
    rpm_search = re.search(r'@(.*)', cell_val)
    if not rpm_search:
        rpm_search = re.search(r'at(.*)', cell_val)

    if rpm_search:
        rpm_part = rpm_search.group(1)
        nums = re.findall(r'\d+', rpm_part)
        if len(nums) == 1:
            rpm = float(nums[0])
        elif len(nums) >= 2:
            rpm = (float(nums[0]) + float(nums[1])) * 0.5

    return pd.Series([torque, rpm])


# Основная функция очистки данных, выполнена по аналогии с ноутбуком
def applying_changes_to_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    df['name'] = df['name'].astype(str).apply(lambda x: x.split()[0])

    df['mileage'] = df['mileage'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else x)
    df['engine'] = df['engine'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else x)
    
    if 'max_power' in df.columns and 4217 in df.index:
        df.loc[4217, ['max_power']] = np.nan # признаю, схитрил. костыль для строки с ошибкой в данных

    df['max_power'] = df['max_power'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else x)

    if "torque" in df.columns:
        df[["torque_Nm", "rpm"]] = df["torque"].apply(magic_parsing)
        df = df.drop(columns=["torque"])
    else:
        if "torque_Nm" not in df.columns:
            df["torque_Nm"] = np.nan
        if "rpm" not in df.columns:
            df["rpm"] = np.nan

    for col in ["engine", "seats"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


@st.cache_resource
def load_model():
    '''
    Загружается модель через pickle и возвращается питоновский объект класса модели
    '''
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_preprocessor_from_train():
    '''
    Воссоздание препроцессинга из обучения модели. Возвращается словарь с объектами и метаданными.
    Надо было обучать модель с пайплайном, но т.к. модель обучалась на результатах StandardScaler + OneHotenc то 
    чтобы корректно делать инферпенс, необходимо применять точно такой же препроцессинг.
    '''
    train_df = pd.read_csv(TRAIN_URL)
    train_df = applying_changes_to_data(train_df)

    # подготовка списков признаков
    num_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in {TARGET_COL, 'seats'}]
    cat_cols = [c for c in train_df.select_dtypes(include=['object']).columns if c != TARGET_COL] + ['seats']
    raw_feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # обрботка числовых признаков 
    num_all = train_df.select_dtypes(include=[np.number]).columns.tolist()
    medians = train_df[num_all].median(numeric_only=True).to_dict()
    train_df[num_all] = train_df[num_all].fillna(medians)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(train_df[num_cols])


    # обрботка категориальных признаков
    
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    X_cat = enc.fit_transform(train_df[cat_cols].astype(str))

    # сборка итоговой матрицы признаков
    X_train_final = sp.hstack([X_num, X_cat], format='csr')

    return {
        'raw_feature_cols': raw_feature_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'medians': medians,
        'scaler': scaler,
        'encoder': enc,
        'train_clean': train_df,
        'X_train_shape': X_train_final.shape
    }


def transform_for_model(df: pd.DataFrame, prep: dict) -> sp.csr_matrix:
    '''
    Преобразование входного датафрейма в матрицу признаков для модели, используя препроцессинг из load_preprocessor_from_train
    '''
    df = df.copy()

    df = applying_changes_to_data(df)

    # Заполняем числовые NaN значениями медиан, посчитанных по train
    for col in prep['num_cols']:
        df[col] = df[col].fillna(prep['medians'].get(col, np.nan))

    # Применяем scaler и encoder и собираем в разреженную матрицу
    X_num = prep['scaler'].transform(df[prep['num_cols']])
    X_cat = prep['encoder'].transform(df[prep['cat_cols']].astype(str))

    X = sp.hstack([X_num, X_cat], format='csr')
    return X



# !!!!!!!!!!!!!!! Интерфейс приложения !!!!!!!!!!!!!!!

st.title('🚗 Car price prediction')

try:
    MODEL = load_model() # загрузка модели
    PREP = load_preprocessor_from_train() # загрузка препроцессинга
except Exception as e:
    st.error(f'Ошибка инициализации: {e}')
    st.stop()

# Сравнение числа признаков после препроцессинга с числом признаков, на которых обучалась модель
n_features_expected = getattr(MODEL, 'coef_', None) # получение числа признаков, на которых обучалась модель
if n_features_expected is not None: # если есть coef_ (линейная модель)
    n_features_expected = int(np.asarray(n_features_expected).shape[0]) # число признаков — длина вектора коэффициентов
    if PREP['X_train_shape'][1] != n_features_expected:
        st.warning(
            f'Ожидается признаков: {n_features_expected}, \n'
            f'Получилось после препроцессинга: {PREP["X_train_shape"][1]}\n'
            'Предобработка не совпала с тем, на чём обучалась модель'
        )


uploaded_file = st.file_uploader('Загрузите CSV с авто', type=['csv'])

if uploaded_file is None:
    st.info('Чтобы получить предсказания загрузите CSV файл')
    st.stop()

df = pd.read_csv(uploaded_file)

# Предсказания по загруженному файлу
try:
    # преобразование данных
    X = transform_for_model(df, PREP)
    y_pred = MODEL.predict(X.toarray() if sp.issparse(X) else X) 
    df_out = df.copy()
    df_out.loc[applying_changes_to_data(df_out.copy()).index, 'predicted_price'] = y_pred
except Exception as e:
    st.error(f'Ошибка при обработке данных: {e}')
    st.stop()


st.subheader('📊 Результаты')

c1, c2, c3 = st.columns(3)
with c1:
    st.metric('Количество авто', len(df_out))
with c2:
    st.metric('Средняя предсказанная цена', f'{df_out["predicted_price"].mean():,.0f} ₽')
with c3:
    st.metric('Медианная предсказанная цена', f'{df_out["predicted_price"].median():,.0f} ₽')

try:
    y_true = pd.to_numeric(df_out[TARGET_COL], errors='coerce')
    mask = y_true.notna()

    y_true_m = y_true.loc[mask].to_numpy()
    y_pred_m = np.asarray(y_pred)[mask.to_numpy()]

    rmse = mean_squared_error(y_true_m, y_pred_m)**0.5
    r2 = r2_score(y_true_m, y_pred_m)

    st.caption(f'RMSE на загруженном файле: {rmse:,.0f}')
    st.caption(f'R² на загруженном файле: {r2:.3f}')
except Exception:
    pass


st.subheader('📈 Визуализации')

fig1 = px.histogram(
    df_out,
    x='predicted_price',
    nbins=40,
    title='Распределение предсказанных цен',
    labels={'predicted_price': 'Цена'},
)
st.plotly_chart(fig1, width="stretch")

if TARGET_COL in df_out.columns:
    # Для корректного отображения scatter-прота трансформируем целевой столбец в число и убираем NaN
    df_sc = df_out.copy()
    df_sc[TARGET_COL] = pd.to_numeric(df_sc[TARGET_COL], errors='coerce')
    df_sc = df_sc.dropna(subset=[TARGET_COL])

    if len(df_sc) > 0:
        fig2 = px.scatter(
            df_sc,
            x=TARGET_COL,
            y='predicted_price',
            title='Предсказанная цена vs реальная',
            labels={TARGET_COL: 'Реальная цена', 'predicted_price': 'Предсказанная цена'},
        )
        # Добавляем ориентирную линию y=x чтобы визуально оценить отклонения предсказаний
        mn = float(min(df_sc[TARGET_COL].min(), df_sc['predicted_price'].min()))
        mx = float(max(df_sc[TARGET_COL].max(), df_sc['predicted_price'].max()))
        fig2.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash='dash'))
        st.plotly_chart(fig2, width="stretch")

# Визуализация весов модели
coef = MODEL.coef_.ravel()

# имена признаков
feature_names = (
    PREP["num_cols"] + list(PREP["encoder"].get_feature_names_out(PREP["cat_cols"]))
)

weights = (
    pd.DataFrame({
        "feature": feature_names,
        "weight": coef
    })
    .assign(abs_weight=lambda x: x["weight"].abs())
    .sort_values("abs_weight", ascending=False)
    .head(50)
)

fig3 = px.bar(
    weights[::-1],
    x="weight",
    y="feature",
    orientation="h",
    title="Топ признаков по весу"
)
st.plotly_chart(fig3, width="stretch")

# Форма для предсказания одного авто
st.subheader('🔮 Предсказание для одного авто')

train_clean = PREP['train_clean']
raw_cols = PREP['raw_feature_cols']

# варианты для selectbox берём из train
brand_options = sorted(train_clean['name'].dropna().astype(str).unique().tolist()) if 'name' in train_clean.columns else ['Unknown']
fuel_options = sorted(train_clean['fuel'].dropna().astype(str).unique().tolist()) if 'fuel' in train_clean.columns else ['Unknown']
seller_options = sorted(train_clean['seller_type'].dropna().astype(str).unique().tolist()) if 'seller_type' in train_clean.columns else ['Unknown']
trans_options = sorted(train_clean['transmission'].dropna().astype(str).unique().tolist()) if 'transmission' in train_clean.columns else ['Unknown']
owner_options = sorted(train_clean['owner'].dropna().astype(str).unique().tolist()) if 'owner' in train_clean.columns else ['Unknown']
seats_options = sorted(train_clean['seats'].dropna().astype(str).unique().tolist()) if 'seats' in train_clean.columns else ['5']

defaults = {k: float(v) for k, v in PREP['medians'].items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

with st.form('one_car_form'):
    left, right = st.columns(2)

    with left:
        name = st.selectbox('name (brand)', brand_options, index=0)
        fuel = st.selectbox('fuel', fuel_options, index=0)
        seller_type = st.selectbox('seller_type', seller_options, index=0)
        transmission = st.selectbox('transmission', trans_options, index=0)
        owner = st.selectbox('owner', owner_options, index=0)
        seats = st.number_input('seats', value=int(defaults.get('seats', 5)), step=1)

    with right:
        year = st.number_input('year', value=int(defaults.get('year', 2015)), step=1)
        km_driven = st.number_input('km_driven', value=float(defaults.get('km_driven', 50000)), step=1000.0)
        mileage = st.number_input('mileage', value=float(defaults.get('mileage', 18.0)), step=0.1)
        engine = st.number_input('engine', value=float(defaults.get('engine', 1200)), step=10.0)
        max_power = st.number_input('max_power', value=float(defaults.get('max_power', 80.0)), step=0.5)
        torque_nm = st.number_input('torque_Nm', value=float(defaults.get('torque_Nm', 120.0)), step=1.0)
        rpm = st.number_input('rpm', value=float(defaults.get('rpm', 2000.0)), step=10.0)

    submitted = st.form_submit_button('Предсказать', use_container_width=True)

if submitted:
    try:
        one = pd.DataFrame([{
            'name': name,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'year': year,
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque_Nm': torque_nm,
            'rpm': rpm,
            'seats': 5,
        }])

        # Трансформируем и предсказываем аналогично batch-режиму
        X_one = transform_for_model(one, PREP)
        pred_one = float(MODEL.predict(X_one.toarray() if sp.issparse(X_one) else X_one)[0])

        st.success(f'Предсказанная цена: **{pred_one:,.0f} ₽**')
    except Exception as e:
        st.error(f'Ошибка при предсказании: {e}')
