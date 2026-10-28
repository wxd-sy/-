' 智能质检台 · 免安装启动脚本（exe 被安全策略拦截时的后备方案）
' 双击即可：静默启动后端并打开浏览器；不常驻托盘，停止请运行「停止服务.bat」
Option Explicit
Dim sh, fso, py, app, url, http, i, ok, scriptDir

Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
py = "C:\ProgramData\anaconda3\envs\yolov8_310\pythonw.exe"
app = scriptDir & "\app.py"
url = "http://127.0.0.1:8765"

If Not fso.FileExists(py) Then
  MsgBox "找不到 Python 环境：" & vbCrLf & py, 16, "智能质检台"
  WScript.Quit 1
End If
If Not fso.FileExists(app) Then
  MsgBox "找不到 app.py：" & vbCrLf & app, 16, "智能质检台"
  WScript.Quit 1
End If

ok = False
On Error Resume Next
Set http = CreateObject("MSXML2.XMLHTTP")
http.Open "GET", url & "/api/health", False
http.Send
If Err.Number = 0 And http.Status = 200 Then ok = True
Err.Clear
On Error GoTo 0

If Not ok Then
  sh.CurrentDirectory = scriptDir
  sh.Run """" & py & """ """ & app & """ --port 8765", 0, False

  For i = 1 To 40
    WScript.Sleep 500
    On Error Resume Next
    Set http = CreateObject("MSXML2.XMLHTTP")
    http.Open "GET", url & "/api/health", False
    http.Send
    If Err.Number = 0 And http.Status = 200 Then ok = True
    Err.Clear
    On Error GoTo 0
    If ok Then Exit For
  Next
End If

If ok Then
  sh.Run url
Else
  MsgBox "后端启动超时，请查看日志：" & vbCrLf & scriptDir & "\logs", 48, "智能质检台"
End If
