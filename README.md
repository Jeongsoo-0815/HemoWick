## HemoWick: Deep Learning for Hemoglobin and Hematocrit Estimation from Time-Series Blood-Spot Images
#### ✔ Python 3.10 is recommended.
#### ✔ If you have any issues, please open an issue or submit a pull request.

## Environment setting

The commands below use an Ubuntu/Linux shell with Python 3.10 installed.

### 1. Install Git
```bash
$ sudo apt-get install git

$ git config --global user.name <user_name>
$ git config --global user.email <user_email>
```

<br>

### 2. Clone this repository to your local path
```bash
$ cd <your_path>
$ git clone https://github.com/Jeongsoo-0815/HemoWick.git
$ cd HemoWick
```

<br>

### 3. Create the virtual environment (optional)

#### - Install the virtual environment package
```bash
$ sudo apt-get install python3.10-venv
```

#### - Create your virtual environment (insert your venv_name)
```bash
$ python3.10 -m venv <venv_name>
```

#### - Activate your virtual environment
```bash
$ source ./<venv_name>/bin/activate
```
**&rarr; Terminal will be...**   ```(venv_name) $ ```

#### - Install the required packages
```bash
$ python -m pip install -r requirements.txt
```

<br>

## Trained weight

- HemoWick_Korea Cohort_weight: [[Google Drive]](https://drive.google.com/file/d/1e0RJbo3p-Y9C1nCSYFkfsQl6AhmZgRVI/view?usp=drive_link)
- HemoWick_Senegal Cohort_weight: [[Google Drive]](https://drive.google.com/file/d/1BGXVsnH45PlbrG-X5CbLeMpe915sAtY_/view?usp=drive_link)
