; Inno Setup Script for Yandex Music Assistant Patch
[Setup]
AppId={{D374B2A5-8C59-4E62-8114-1FB5CD51A293}
AppName=Yandex Music Assistant Patch
AppVersion=1.2.0
AppPublisher=iReink
DefaultDirName=C:\YandexMusicAssistant
DirExistsWarning=no
EnableDirDoesntExistWarning=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=installer_output
OutputBaseFilename=YandexMusicAssistant_Patch
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist_app\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\voice.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\kws_engine.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\aec_engine.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\tts.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\player.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\config.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\parser.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\llm_nlu.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\live_session.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\env_manager.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\search_service.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\service.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\text_normalizer.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\logger.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\Launch.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist_app\sounds\*"; DestDir: "{app}\sounds"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "dist_app\models\kws\*"; DestDir: "{app}\models\kws"; Flags: ignoreversion recursesubdirs createallsubdirs
