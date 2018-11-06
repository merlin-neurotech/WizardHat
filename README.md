# WizardHat
![logo](https://github.com/merlin-neurotech/WizardHat/blob/master/WizardHatLogoSmall.jpg)

WizardHat is a library for the streaming and handling of EEG data from consumer-grade devices using the Lab Streaming Layer (LSL) protocol. WizardHat's prupose is to enable users and especially first timers to flexibly build brain-computer interfaces (BCIs) without the fuss of configuring a streaming environment. WizardHat was built by Merlin Neurotech at Queen's University. Currently, WizardHat supports the Muse (2016) brain-sensing headband, the OpenBCI Ganglion, and runs on Python 3.6. WizardHat is easy to use and only requires three lines of code to get started. WizardHat's framework enables streaming, manipulation, and visualization of online EEG data.

For first time python users, please refer to our [beginner's guide](https://docs.google.com/document/d/1q9CNgSgUsNCRasLZtZ7D-2JpB7OcNvSsS3X1a1zHK-U/edit?usp=sharing) on how to install everything from scratch. WizardHat's documentation can be found [here](https://docs.google.com/document/d/1dOymsVdVxN3SgN3mRIzHV1xmjpIjEvz5QSDIQ66D6To/edit?usp=sharing).

## Note : Active Development
Our dedicated team at Merlin Neurotech is continously working to improve WizardHat and add new functionality.
Current on-going projects:
- MNE Library Compatibility
- Implementing simple filters

Check back soon if the feature you are looking for is under development!

## Getting Started

The procedure for installing WizardHat depends on whether or not you will be contributing to its development. In either case, begin by creating and activating a new python virtual environment.

### Installing for use only
Simply run

	pip install wizardhat

This will automatically install the most recent release of WizardHat along with the required dependencies.

### Installing for development
To set up WizardHat for development, begin by forking the repository on GitHub, then clone your fork:

	git clone https://github.com/<your-github-username>/WizardHat.git

If you are also developing for ble2lsl, fork and then clone the ble2lsl repository as well, and install its dependencies:

	git clone https://github.com/<your-github-username>/ble2lsl.git
	cd ble2lsl
	pip install -r requirements.txt
	pip install -e .
	cd ..

Whether or not you cloned ble2lsl, install the remaining dependencies for WizardHat:

        cd WizardHat
	pip install -r requirements.txt
	pip install -e .

The last command creates an editable install of WizardHat; i.e. if you `import wizardhat` in any terminal/script while your WizardHat virtual environment is activated, it will import the current version of the modules from your fork's folder (i.e. the imported module will reflect your developments).

### Finally

For more details on how to set up your Python environment on Windows/MacOS/Linux please refer to our detailed instructions in the documentation file.

Next, to ensure a bug free experience, open [your virtual env name]/lib/python3.6/site packages/pygatt/backends/bgapi/bgapi.py in a text or code editor and add:

	time.sleep(0.25)

between line 200 and 201 and save the file. This ensures that the Bluetooth protocol will be given adequate time to connect to the Muse before timing out.

Now you are ready to use WizardHat!

## Working with WizardHat

Our library provides two options when building, debugging, or testing your BCI code:

1) Streaming data from Muse or other EEG device
2) Streaming randomly generated data for testing and debugging

To begin streaming, you will need first to import `ble2lsl` and `wizardhat.acquire` into your Python workspace. The BLE device parameters for different devices are stored in respective modules in `ble2lsl.devices`.

	import ble2lsl
	from ble2lsl.devices import muse2016
	from wizardhat import acquire

You then need to create a streaming outlet which establishes a Bluetooth connection with the EEG device:

	streamer = ble2lsl.Streamer(muse2016)

To stream dummy data through an outlet that mimics (number of channels, sample rate, and metadata) the Muse 2016

	dummy_streamer = ble2lsl.Dummy(muse2016)

Next, to store and record the data, add the following line to capture the outlet stream:

	receiver = acquire.Receiver()

Notice how you do not need to pass the outlet streamer as an argument to this function. LSL can stream over a local network, and `ble2lsl` need not be run in the same process as `wizardhat`. LSLStreamer automatically finds and connects to the LSL outlet. (More work is needed to allow LSLStreamer to distinguish multiple outlets, when they are available.)

Now that your streamer is is receiving data, you are able to visualize and manipulate it online. The data object is a structured array with timestamps as the first column, and EEG channel data values as the following columns. It contains both raw values and metadata regarding the device in use. The current copy of the stored data is in

	streamer.buffers["EEG"].data

After each time window, data is saved to a CSV file in your directory under a folder called 'data' and is constantly updated while your stream is running. Each new streaming session (specifically, data object) you establish will create a new CSV file, accompanied by a JSON file of the same named containing the stream metadata.

These are the basics of WizardHat; to learn how to transform, filter, and visualize the data on a graph, refer to the [WizardHat documentation](https://docs.google.com/document/d/1dOymsVdVxN3SgN3mRIzHV1xmjpIjEvz5QSDIQ66D6To/edit?usp=sharing).

To gain a deeper understanding into how our framework operates, take a look under the hood.

## Authors
Matt Laporte,
Ben Cuthbert,
Omri Nachmani

## Contributors
Abigail,
Jorge,
Dan,
Colton,
Teghan,
Chris,
Hamada

## Acknowledgements
This project was inspired by Alexander Barachant's [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) from which some of the modules were originally based. The device specification for the OpenBCI Ganglion is largely derived from [OpenBCI_Python](https://github.com/OpenBCI/OpenBCI_Python).
