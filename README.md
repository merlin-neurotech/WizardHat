# WizardHat
![logo](https://github.com/merlin-neurotech/WizardHat/blob/master/WizardHatLogoSmall.jpg)

WizardHat is a library for handling, transforming, and visualizing EEG data streamed from consumer-grade devices over the Lab Streaming Layer (LSL) protocol. WizardHat's purpose is to enable users to easily and flexibly build brain-computer interfaces (BCIs) using a high-level API, with technical details managed in the background. Paired with ble2lsl, WizardHat currently supports streaming from the Muse (2016) brain-sensing headband and the OpenBCI Ganglion. WizardHat is easy to use and only requires three lines of code to get started.

WizardHat was built by Merlin Neurotech at Queen's University.

## Note : Active Development
Our dedicated team at Merlin Neurotech is continously working to improve WizardHat and add new functionality.
Current on-going projects:
- MNE Library Compatibility
- Implementing simple filters

Check back soon if the feature you are looking for is under development!

## Getting Started

The procedure for installing WizardHat depends on whether or not you will be contributing to its development. In either case, begin by creating and activating a new Python virtual environment.

### Installing for use only
Simply run

	pip install wizardhat

This will automatically install the most recent release of WizardHat along with the required dependencies, including ble2lsl.

### Installing for development
To set up WizardHat for development, begin by forking the repository on GitHub, then clone your fork:

	git clone https://github.com/<your-github-username>/WizardHat.git

If you are also developing for ble2lsl, fork and then clone the ble2lsl repository as well, and install its dependencies:

	git clone https://github.com/<your-github-username>/ble2lsl.git
	cd ble2lsl
	pip install -e .
	cd ..

Whether or not you cloned ble2lsl, install the remaining dependencies for WizardHat:

        cd WizardHat
	pip install -e .

The last command creates an editable install of WizardHat; i.e. after installation into an active virtual environment, running `import wizardhat` in any Python terminal/script will import the current (i.e. working) version of the modules from your fork's folder. 

### Finally
To ensure a bug free experience, open [your virtual env name]/lib/python3.6/site packages/pygatt/backends/bgapi/bgapi.py in a text or code editor and add:

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

To stream dummy data through an outlet that mimics (number of channels, sample rate, and metadata) the Muse 2016:

	dummy_streamer = ble2lsl.Dummy(muse2016)

Next, to store and record the data, add the following line to capture the outlet stream:

	receiver = acquire.Receiver()

Notice how you do not need to pass the outlet streamer as an argument to this function. LSL can stream over a local network, and `ble2lsl` doesn't need to run in the same process as `wizardhat`. Instead, `Receiver` automatically finds and connects to the LSL outlet; if multiple outlets are available, it will prompt you to choose from a list.

Now that your streamer is receiving data, you can visualize and manipulate it online. Data is streamed into structured arrays with timestamps in their first columns and channel data in their remaining columns. By default, these arrays contain the last 10 s (to the nearest sample) of streamed data, and are accessed through the `streamer.buffers` dictionary. For example, the structured array of most recent timestamps and samples for the Muse 2016's EEG stream (labeled "EEG") is accessed by:

	streamer.buffers["EEG"].data

The available streams depend on the device, and are specified in the device's module in `ble2lsl.devices`. For example, the Muse 2016 headband provides the following stream types: "EEG", "accelerometer", "gyroscope", "telemetry", and "status". After each time window, data is saved to a CSV file in your directory under a folder called 'data' and is constantly updated while your stream is running. Each data buffer has an associated CSV file, accompanied by a JSON file of the same name providing metadata.

To gain a deeper understanding into how our framework operates, take a look under the hood.

## Acknowledgements
This project was inspired by Alexander Barachant's [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) from which some of the modules were originally based. The device specification for the OpenBCI Ganglion is largely derived from [OpenBCI_Python](https://github.com/OpenBCI/OpenBCI_Python).
