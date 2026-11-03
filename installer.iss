; Inno Setup Script for Yandex Music Assistant
#define MyAppName "Яндекс.Музыка Ассистент"
#define MyAppVersion "1.0"
#define MyAppPublisher "Yandex Music Assistant Team"
#define MyAppExeName "Launch.bat"

[Setup]
AppId={{D37E6F10-6C25-4A9B-98F1-4F44828E43F7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\YandexMusicAssistant
DefaultGroupName={#MyAppName}
OutputDir=C:\Users\Reink\.gemini\antigravity\scratch\yandex_music_assistant\installer_output
OutputBaseFilename=YandexMusicAssistant_Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=yes

CloseApplications=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Настройки пользователя и токен авторизации не перезаписываются при обновлении поверх
Source: "dist_app\settings.json"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "dist_app\.env"; DestDir: "{app}"; Flags: onlyifdoesntexist
; Все исполняемые файлы, скрипты, рантайм и модели обновляются до актуальной версии
Source: "dist_app\*"; DestDir: "{app}"; Excludes: "settings.json,.env,*.log"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: shellexec postinstall skipifsilent nowait
