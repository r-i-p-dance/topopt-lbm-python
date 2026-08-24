"""
cd C:/CONSANARCHY/Warwick/URSS/topopt-lbm-python

pip uninstall lbm -y
pip uninstall topopt -y

# Remove folder recursively:
Remove-Item -Recurse -Force "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages\lbm"
Remove-Item -Recurse -Force "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages\topopt"
Remove-Item "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages\__editable__.lbm*" -ErrorAction SilentlyContinue
Remove-Item "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages\__editable__.topopt*" -ErrorAction SilentlyContinue
Remove-Item "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages\lbm-*.dist-info" -Recurse -ErrorAction SilentlyContinue
Remove-Item "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages\topopt-*.dist-info" -Recurse -ErrorAction SilentlyContinue

# 4. Verify site-packages is clean
Get-ChildItem "C:\Users\stsvi\AppData\Local\Programs\Python\Python312\Lib\site-packages" | Where-Object { $_.Name -like "*lbm*" -or $_.Name -like "*topopt*" }

pip install -e ..\lbm-2d-python
pip install -e .

python -c "import lbm; print(lbm.__file__); print(lbm.__path__)"
python -c "import topopt; print(topopt.__file__); print(topopt.__path__)"
"""