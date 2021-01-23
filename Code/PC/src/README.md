# AcCAPPCHA implementation for PC
## Dependencies installation
<details><summary><b>PyAudio</b></summary>
<b><i>Linux:</i></b><br>
  <code>
  sudo apt install portaudio19-dev
  pip3 install pyaudio
  </code><br><br>
  <b><i>Windows:</i></b><br>
  Check the version and either you have 64 or 32 Python just open python on terminal, obtaining for example this result:<br>
  <img src="img/version_python.PNG" width="650" alt="version_python"><br>
  Download from the appropriate <i>.whl</i> file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio). An example of the name of this file is <b>PyAudio‑0.2.11‑cp37‑cp37m‑win_amd64.whl</b><br><br>
  Then go to the download folder and install it through the command:<br>
  <code>
  pip3 install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
  </code><br>
  or<br>
  <code>
  python3 -m pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
  </code>
</details>
<details><summary><b>Tensorflow</b></summary>
  Run the following command on terminal:<br>
  <code>
  pip3 install tensorflow
  </code><br>
  or<br>
  <code>
  python3 -m pip install tensorflow
  </code><br>
  Instead of installing <i>tensorflow</i>, I installed <i>tensorflow-gui</i> on Windows to exploit the computation power of my GPU Nvidia GTX 1050 Ti.<br><br>
  <b><i>Linux:</i></b><br>
  I needed to explicitly install keras after tensorflow, using:
  <code>
  pip3 install keras
  </code><br><br>
  <b><i>Windows:</i></b><br>
  Before running the previous command on cmd (as administrator) you need to manage MAX_PATH limitations
  of Windows. To do so, you need to set the register key <code>Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled</code>
  to value <i>1</i>.
</details>
<details><summary><b>Install CUDA and CUDNN</b></summary>
This dipendency is important for tensorflow to perform computation using the user's NVIDIA GPU card. You can follow the [official installation guide](https://www.tensorflow.org/install/gpu) made by tensorflow team.
</details>
<details><summary><b>Other modules</b></summary>
  Type the following command on terminal:<br>
  <code>
  pip3 install matplotlib pyaudio scipy numpy wave pynput Datetime termcolor argparse csv colorama
  </code><br>
  or<br>
  <code>
  python3 -m pip install matplotlib pyaudio scipy numpy wave pynput Datetime termcolor argparse csv colorama
  </code>
</details>

<code>
python3 NeuralNetwork.py
</code>

<details><summary><a href="AcquireAudio.py"><i><b>AcquireAudio.py</b></i></href></summary>
  File with <i>AcquireAudio<i> class definition that is used to create an object for audio recording and key logging in parallel.
</details>
<details><summary><a href="AcquireAudio.py"><i><b>ExtractFeatures.py</b></i></href></summary>File with <i>ExtractFeatures<i> class definition that is used to create an object for the analysis and extraction of an audio signal.
</details>
<details><summary><a href="AcquireAudio.py"><i><b>PlotExtract.py</b></i></href></summary>File with <i>PlotExtract<i> class definition that is used to plot or extract features from audios in an input folder.
</details>
<details><summary><a href="AcquireAudio.py"><i><b>DatasetAcquisition.py</b></i></href></summary>Main file with command line arguments and creation of objects of other classes.
</details>
  <details><summary><a href="AcquireAudio.py"><i><b>NeuralNetwork.py</b></i></href></summary>File with <i>NeuralNetwork<i> class definition that is used to create an object for construction of a neural network for training and test phase of the algorithm.
</details>
