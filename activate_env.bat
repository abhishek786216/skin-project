@echo off
echo Activating virtual environment for skin cancer detection project...
cd /d "c:\Users\abhis\OneDrive\Desktop\skin"
call venv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo You can now run:
echo   python simple_app.py
echo   streamlit run simple_app.py
echo   python quick_train_model.py
echo.
cmd /k