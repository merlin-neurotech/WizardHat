# Wizard Hat

Wizard Hat is library for the streaming and handling of EEG data from consumer-grade devices using the Lab Streaming Layer (LSL) protocol. Wizard Hat's prupose is to enable users to flexibly build brain-computer interfaces (BCIs) without the fuss of configuring a streaming environment. Currently, Wizard Hat supports the Muse (2016) brain-sensing headband and runs on Python 3.6. Wizard Hat is easy to use and only requires three lines of code to get started. Wizard Hat's framework enables streaming and manipulation of online EEG data, visualization, and filtering using the MNE library. 

## Getting Started

To set up Wizard Hat, begin by cloning this repository on your local environment. Once cloned, ensure you are in a new virtual environment and download the required depencies.

	pip install -r requirements.txt

For more details on how to set up your python environment on Windows/macOS/Linux please refer to our detailed instructions in the documentation file.

Next, to ensure a bug free experience, open [your virtual env name] ->lib ->python3.6 ->site packages -> pygatt -> backends ->bgpai -> bgapi.py in a text or code editor and add:

	time.sleep(0.25)

between line 200 and 201 and save the file. This ensures that the bluetooth protocol will be given adequate time to connect to the muse before timing out.

Now you are ready to use Wizard Hat!

## Working with Wizard Hat

Our library provides two options when building, debugging, or testing your BCI code:

1)Streaming data from Muse or other EEG device
2)Streaming randomly generated data for debugging purposes

To begin streaming, you will ned first to import Wizard Hat into your python workspace

	import WizardHat as wiz

You then need to create a streaming outlet which establishes a bluetooth connection with the EEG device

	outlet = wiz.ble2lsl.LSLOutletStreamer()

To stream dummy data for debugging purpose

	dummy_outlet = wiz.ble2lsl.LSLOutletDummy()

You should then see a "Connected" message printed to your console indicating that either the EEG device or the dummy outlet are streaming data.

Next, to store and record the data, add the following line to capture the outlet stream:

	streamer_inlet = acquire.LSLStreamer()

Notice how you do not need to pass the outlet streamer as an argument to this function, as LSLStreamer automatically finds and connects to this stream.

Now that your streamer is is receiving data, you are able to visualize and manipulate it online within a time window of your choosing (defaults to 10). The data object is a structured array with timestamps as the first column, and EEG channel data values as the following columns. It contains both raw values and metadata regarding the device in use. To view this

	print(streamer_inlet.data.data)

After each time window, data is saved to a csv file in your directory under a folder called 'data' and is constantly updated while your stream is running. Each new streaming session you establish will create a new csv file. 

These are the basics of Wizard Hat, to learn how to transform, filter, and visualize the data on a graph, refer to the documentation file in this repository.

To gain a deeper understanding into how our framework operates, take a look under the hood. 

## Authors
Matt Laporte 
Ben Cuthbert
Omri Nachmani

## Contributors 
Jorge
Dan
Colton
Teghan
Chris
Hamada

## Acknowledgements 
This project was inspired by Alexander Barachant's muse-lsl and contains several lines of code and files written by him. When we first got started with the Muse, we were inspired to recreate his work to fit our needs. As our project grew larger and larger we came to realize it can now be considered an independent library that may be of use to others who are working on BCI and EEG related work.
