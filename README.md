# GAN-and-CNN
## Conditional GANの学習・画像生成
Conditinal GANを学習させる場合は`cgan/run_mnist_train.sh`を，画像を生成する場合は`cgan/run_mnist_test.sh`を実行する．
学習完了時，各ラベルにつき338枚のサンプル画像を生成してくれるので，追加のサンプルが必要でなければ`run_mnist_train.sh`を実行するだけでよい．生成したサンプル画像は`cgan/samples/batch_image`に保存される．  
## 生成された画像の分割
今回のConditional GANのプログラムではバッチサイズ分の画像が1枚の画像として結合して出力される．`split_images.ipynb`は，これを1枚ずつmnistの画像サイズ（28×28）に切り分けるプログラム．`cgan/samples/batch_image`内の画像を切り分けて`generated_data`に保存する．  
## CNNの学習  
CNNの学習は`mnist_cnn/mnist-cnn.py`を実行する．この際，ソースコード中96行目の`waights_file`に与えるパスを存在しないファイルに設定すると学習を行う．既に存在しているファイルが指定されている場合は，指定されたファイルを読み込んで精度の計算のみを行う．学習中の損失関数や精度の値は`mnist_cnn_fig`内にグラフ画像として保存される．  
## CNNによるエントロピーの計算
エントロピーの計算は`mnist_cnn/compute_prob.py`を実行する．このプログラムは`generated_data`内の画像を読み込み，ソースコード中71行目で指定されたファイルに保存されている重みを用いてエントロピーを計算する．エントロピーの計算結果は`entropy_list.csv`に保存される．
## 計算結果の視覚化
エントロピーの度数分布やサンプル画像と対応する確率のグラフ表示は`Graph.ipynb`を用いる．
