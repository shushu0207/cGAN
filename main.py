# main.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --- モデル定義 ---
# 提供されたコードからGeneratorクラスを転記
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- 定数とデバイス設定 ---
LATENT_DIM = 10
N_CLASSES = 10
MODEL_PATH = 'generator.pth' # 保存したモデルファイルのパス

# 使用デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
# キャッシュを利用して、アプリの再実行時にモデルを再ロードしないようにする
@st.cache_resource
def load_model():
    """
    保存された.pthファイルを読み込み、Generatorモデルを返す関数
    """
    model = Generator(LATENT_DIM, N_CLASSES).to(device)
    try:
        # weights_only=True を指定して安全に重みファイルをロードする
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()  # 推論モードに設定
        return model
    except FileNotFoundError:
        st.error(f"モデルファイル '{MODEL_PATH}' が見つかりません。")
        st.info("学習済みモデルの重みファイルを同じディレクトリに配置してください。")
        return None
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        return None

generator = load_model()

# --- Streamlit UI ---
st.title('🔢 手書き数字画像生成アプリ (Conditional GAN)')
st.write('サイドバーで生成したい数字を選んでください。')

with st.sidebar:
    st.header('コントロールパネル')
    selected_digit = st.selectbox(
        '生成したい数字を選択:',
        options=list(range(N_CLASSES)),
        index=0
    )
    generate_button = st.button('画像を生成', type="primary")

# --- 画像生成と表示 ---
if generate_button and generator is not None:
    with st.spinner('画像を生成中です...'):
        # 1. ノイズとラベルの準備
        z = torch.randn(1, LATENT_DIM, device=device) # 1枚だけ生成
        label = torch.LongTensor([selected_digit]).to(device)

        # 2. 推論の実行
        with torch.no_grad():
            generated_img_tensor = generator(z, label)

        # 3. 表示用に画像を変換
        # [-1, 1] の範囲から [0, 1] の範囲に正規化し、NumPy配列に変換
        img_np = generated_img_tensor.squeeze().cpu().numpy()
        img_np = (img_np + 1) / 2.0

        # 4. 結果の表示
        st.subheader(f'「{selected_digit}」の生成結果')
        st.image(img_np, caption=f'生成された数字: {selected_digit}', width=200)

elif generator is None:
    st.warning("モデルがロードされていないため、処理を実行できません。")
