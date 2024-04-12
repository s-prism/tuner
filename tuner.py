import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import pyaudio 
import pyfftw
import math
import sounddevice as sd
import time

def find_base(base,new_freqs,amps,old_divisor=1):
    if len(new_freqs>0):
        base=base*old_divisor/(old_divisor+1)
        new_freqs=new_freqs[new_freqs<(6*base)]
        temp=(new_freqs%base)/base
        harm=np.logical_or(temp>0.9,temp<0.1) 

        harm_index=np.squeeze(np.argwhere(harm))

        base_changed=np.all(amps[harm_index]>(6*np.min(amps))) 
        new_freqs=np.delete(new_freqs,harm_index)     
        amps=np.delete(amps,harm_index)

        if (not base_changed):
            if old_divisor<4:
                base=find_base(base,new_freqs,amps,old_divisor+1)
            else:
                base=base*(old_divisor+1)
    return base

def get_dominant_frequency(freqs,amps):
    base=freqs[np.argmax(amps)]
    new_freqs=freqs[freqs<(6*base)]
    new_freqs=new_freqs[new_freqs>(base/4)]

    temp=(new_freqs%base)/base
    harm_index=np.squeeze(np.argwhere(np.logical_or(temp>0.88,temp<0.12))) #change values of temp, or look at weighting based on amplitude
    new_freqs=np.delete(new_freqs,harm_index)                              #or experiment with changing line 54
    amps=np.delete(amps,harm_index)

    base=find_base(base,new_freqs,amps)
    if base<20:
        base=20.0
    return base

            

def find_frequency(fft_result):
    fft_freq = np.linspace(0,44100/2-1,(len(fft_result))) 

    f=np.array(fft_result,dtype=np.complex128)

    #get rid of negligible frequencies
    f=f[fft_freq>=20]
    fft_freq=fft_freq[fft_freq>=20]

    notable_power=np.max(f)/8    #8
    f[f<notable_power]=0

    #frequency of signal is the first local maximum point
    if len(f[f>notable_power])!=0:
        larger_than_next=np.squeeze(np.argwhere(f[:-1]>f[1:]))
        larger_than_prev=1+np.squeeze(np.argwhere(f[1:]>f[:-1]))
        max_point=np.intersect1d(larger_than_next,larger_than_prev)

        if len(max_point)==0:
            freq_to_return=20.0 
        else:
            freq_to_return=get_dominant_frequency(fft_freq[max_point],f[max_point])
    else:
        freq_to_return=20.0 

    return freq_to_return


def get_note(freq,a4):
    x= 4 + math.log2(freq/a4)

    flat=u"\u266D"
    noteArray=['A',('A#','B'+flat),'B','C',('C#','D'+flat),'D',('D#','E'+flat),'E','F',('F#','G'+flat),'G',('G#','A'+flat)]
    note = noteArray[round(12*x)%12]
    octave = str(int((12*x + 9)//12)) 

    if len(note)==1:
        note_and_octave = note + octave  
    else:
        note_and_octave = note[0] + octave + ' / ' + note[1] + octave

    z=(12*x)%12
    cents_away= round((z - round(z)) * 100)

    return note_and_octave,cents_away


def sample_to_output(sample,a4=440):
    pyfftw.interfaces.cache.enable()
    fft_data=pyfftw.interfaces.numpy_fft.rfft(sample,64*len(sample)) #4
    freq=find_frequency(fft_data)
    note, cents_away = get_note(freq,a4)
    return freq, note, cents_away

result=None,None,None
a4=440
def callback(in_data, frame_count, time_info, status):
    # If len(data) is less than requested frame_count, PyAudio automatically
    # assumes the stream is finished, and the stream stops.
    global result
    global a4
    result=sample_to_output(np.frombuffer(in_data, dtype=np.float32),a4) #dtype=np.int16
    return (in_data, pyaudio.paContinue)

#####################################################################################

# Define the window layout
col_freq=sg.Text(key="-FREQUENCY-",size=(7,1),justification='right')
col_hz=sg.Text("hertz",size=(13,1),justification='left')
col_cents_away=sg.Text(key="-CENTS-AWAY-",size=(7,1),justification='right')
col_cents=sg.Text("cents",size=(7,1),justification='left')
layout = [
    [sg.Text("Tuner")],
    [sg.Text(key="-NOTE-")],
    [sg.Frame("",
        [
            [sg.Column([[col_freq,col_hz,col_cents_away,col_cents]],pad=(0,0),expand_x=True,expand_y=True)], 
            ],
        )],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Text("Set A4: "),sg.Input(key='-A4-', enable_events=True,default_text="440",size=(4,2),justification='centre')]
]

# Create the form and show it without the plot
window = sg.Window(
    "Tuner",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Helvetica 36",
)
##############################################################
matplotlib.use("TkAgg")

# Setup channel info
device_info=sd.query_devices(sd.default.device,'input')
FORMAT = pyaudio.paFloat32 #pyaudio.paInt16 # data type formate
CHANNELS = 1
RATE = int(device_info['default_samplerate']) 
CHUNK = 2**13 # Block Size (2 bytes per sample i.e. \x01\x00)

# Startup pyaudio instance
audio = pyaudio.PyAudio()
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)
# Record while stream active

x=np.array([0]*20)
y=np.arange(20)
back=-1
fig,ax=plt.subplots(figsize=(10,8),dpi=100)

###
# Link matplotlib to PySimpleGUI Graph
canvas = FigureCanvasTkAgg(fig, window['-CANVAS-'].Widget)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=0)
###

time.sleep(0.1)
# Create an event loop
while stream.is_active():
    ax.cla()
    freq,note,cents_away=result 

    back=(back+1)%20
    x[back]=cents_away
    cents_sign_before='+' if cents_away>=0 else ''

    # Add the plot to the window
    plt.axvline(x=0,color='#075431',linewidth=3)
    ax.plot(np.concatenate((x[back+1:],x[:back+1])),y,color='gray',linewidth=3) 
    ax.set_xlim([-50,50])
    ax.set_ylim([0,20])
    ax.axvspan(-15, 15, alpha=0.2,color='green')
    window['-NOTE-'].update(note)
    window['-FREQUENCY-'].update(str(round(freq,1)))
    window["-CENTS-AWAY-"].update(cents_sign_before + str(cents_away))
    plt.axis('off')
    fig.canvas.draw() 

    event, values = window.read(20)
    # End program if user closes window or
    # presses the OK button
    if event == sg.WIN_CLOSED:
        break
    if event == '-A4-' and values['-A4-']:
        try:
            a4 = int(values['-A4-'])
        except:
            values['-A4-']=""

window.close()

# Stop Recording
stream.stop_stream()
stream.close()
audio.terminate()