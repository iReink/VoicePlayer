; Inno Setup Script for Yandex Music Assistant Patch
[Setup]
AppId={{D374B2A5-8C59-4E62-8114-1FB5CD51A293}
AppName=Yandex Music Assistant Patch
AppVersion=1.1.0
AppPublisher=iReink
DefaultDirName={autopf}\YandexMusicAssistant
DirExistsWarning=no
EnableDirDoesntExistWarning=yes
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=installer_output
OutputBaseFilename=YandexMusicAssistant_Patch
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist_app\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\player.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\voice.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\tts.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\env_manager.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\live_session.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\Launch.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "patch_temp\wheels\scipy-1.18.1-cp313-cp313-win_amd64.whl"; DestDir: "{app}\_patch_wheels"; Flags: ignoreversion deleteafterinstall

[Run]
Filename: "{app}\runtime\python.exe"; Parameters: "-m pip install ""{app}\_patch_wheels\scipy-1.18.1-cp313-cp313-win_amd64.whl"" --no-deps --no-warn-script-location --break-system-packages"; StatusMsg: "Обновление аудио-компонентов синтеза речи (scipy)..."; Flags: runhidden
