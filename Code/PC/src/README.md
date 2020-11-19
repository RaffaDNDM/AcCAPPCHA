# AcCAPPCHA implementation for PC

## Definition of training set
### Dependencies installation
<details><summary><b>Install PyAudio</b></summary>
  <b><i>Linux:</i></b><br>
  <code>
  sudo apt install portaudio19-dev
  pip3 install pyaudio
  </code><br>
  <b><i>Windows:</i></b><br>
  Check the version and either you have 64 or 32 Python just open python on terminal, obtaining for example this result:<br>
  <img src="img/version_python.PNG" width="650" alt="version_python"><br>
  Download from the appropriate <i>.whl</i> file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio). An example of the name of this file is <b>PyAudio‑0.2.11‑cp37‑cp37m‑win_amd64.whl</b><br>
  Then go to the download folder and install it through the command:<br>
  <code>
  pip3 install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
  </code><br>
  or<br>
  <code>
  python3 -m pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
  </code><br>
</details>
<details><summary><b>Install other modules</b></summary>
  Type the following command on terminal:<br>
  <code>
  pip3 install matplotlib pyaudio scipy numpy wave pynput Datetime termcolor argparse csv
  </code><br>
  or<br>
  <code>
  python3 -m pip install matplotlib pyaudio scipy numpy wave pynput Datetime termcolor argparse csv
  </code>
</details>

### Relative code
<details><summary><b>Python Files</b></summary>
  <ul>
  <li><i><b>AcquireAudio.py</b></i><br>File with <i>AcquireAudio<i> class definition that is used to create an object for audio recording and key logging in parallel.</li>
  <li><i><b>ExtractFeatures.py</b></i><br>File with <i>ExtractFeatures<i> class definition that is used to create an object for the analysis and extraction of an audio signal.</li>
  <li><i><b>PlotExtract.py</b></i><br>File with <i>PlotExtract<i> class definition that is used to plot or extract features from audios in an input folder.</li>
  <li><i><b>DatasetAcquisition.py</b></i><br>Main file with command line arguments and creation of objects of other classes.</li>
  </ul>
</details>
<details><summary><b>Commands to launch the program</b></summary>
  <code>
  python3 -m pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
  </code><br>
</details>


## Neural Network creation and training phase