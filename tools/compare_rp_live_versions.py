#!/usr/bin/env python3
"""四通道石头/布实时代码通用启动器。Python>=3.9，仅使用标准库。

放到代码目录。run_rp_binary*.py和07_live_rp_binary*.py均可选择。
选run入口后，还可指定实际07文件；临时绑定，不修改旧脚本或模型。
同一次--run始终使用相同daily_model和detector；每次结果独立保存。
支持--list、--script 文件名 --mode 模式名；每次关闭子程序窗口后返回选择菜单。
"""
from __future__ import annotations

import argparse
import ast
import csv
import contextlib
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import types

DEFAULT_RUN = None
SUMMARY_FIELDS = ["started_at","script","version","script_sha256","mode","trigger_mode",
    "model_sha256","detector_sha256","inputs_unchanged","exit_code","recorded_actions",
    "correct","wrong","manual_misses","accuracy","stone_correct","stone_total",
    "paper_correct","paper_total","result_dir","entry_kind","live_script","live_version","live_sha256"]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_metadata(path):
    values={};defaults={}
    try:
        tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                for target in node.targets:
                    if isinstance(target,ast.Name):values[target.id]=node.value.value
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args:
                try:
                    flag=ast.literal_eval(node.args[0])
                    for key in node.keywords:
                        if key.arg=='default':defaults[flag]=ast.literal_eval(key.value)
                except (ValueError,TypeError,SyntaxError):pass
    except (OSError,UnicodeError,SyntaxError):pass
    version=next((str(values[key]) for key in ['LIVE_VERSION','__version__','VERSION'] if key in values),'未标版本')
    return version,defaults


def modes_for(candidate):
    flags=candidate['flags'];choices=candidate['choices'];defaults=candidate['defaults']
    default_args=[];label='本版本默认配置'
    if defaults.get('--activity-mode')=='guided' or ('guided' in choices.get('activity-mode',[]) and '--activity-mode' not in defaults):
        safe=next((v for v in ['rest','legacy','robust'] if v in choices.get('activity-mode',[])),None)
        if safe:
            default_args=['--activity-mode',safe]
            label='自动配置（guided切换为%s，不预选动作）'%safe
    effective_activity=default_args[-1] if default_args else defaults.get('--activity-mode')
    modes=[{'key':'default','label':label,'args':default_args}]
    for value,label in [('legacy','原始活动门控 legacy'),('robust','稳健静息门控 robust'),('rest','静息门控 rest（本版本默认阈值）')]:
        if value in choices.get('activity-mode',[]) and effective_activity!=value:
            modes.append({'key':value,'label':label,'args':['--activity-mode',value]})
    threshold_flags={'activity-on-z','activity-off-z'}.issubset(flags)
    if 'rest' in choices.get('activity-mode',[]) and threshold_flags:
        common=['--activity-mode','rest','--activity-on-z','4','--activity-off-z','3']
        modes.append({'key':'rest4','label':'自动静息模式，启动4 / 释放3','args':common})
        if 'responsive' in choices.get('live-profile',[]):
            modes.append({'key':'responsive','label':'自动响应模式，启动4 / 释放3','args':common+['--live-profile','responsive']})
    if 'manual' in choices.get('trigger-mode',[]) and 'manual-duration-seconds' in flags:
        modes.append({'key':'manual','label':'按钮模式：点击后做动作，4秒结束判断','args':['--trigger-mode','manual','--manual-duration-seconds','4']})
    return modes


def probe(path):
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    result=subprocess.run([sys.executable,str(path),'--help'],cwd=str(path.parent),env=env,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    help_text=result.stdout+'\n'+result.stderr
    if result.returncode:
        raise ValueError('--help执行失败：'+help_text.strip()[-400:])
    kind='wrapper' if re.search(r'\{[^}\n]*\blive\b[^}\n]*\}',help_text) else 'direct'
    if kind=='wrapper':
        result=subprocess.run([sys.executable,str(path),'live','--help'],cwd=str(path.parent),env=env,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
        if result.returncode:raise ValueError('入口的live --help执行失败：'+result.stderr[-300:])
        help_text=result.stdout+'\n'+result.stderr
    flags=set(re.findall(r'(?<![\w-])--([a-z0-9-]+)\b',help_text))
    required={'run','stream-name','output'} if kind=='wrapper' else {'model','detector','stream-name','output'}
    if not required.issubset(flags):
        return None,('流程入口的live命令缺少--run、--stream-name或--output参数'
            if kind=='wrapper' else '不是接受--model和--detector的实时代码')
    choices={key:[v.strip() for v in value.split(',')] for key,value in
        re.findall(r'--([a-z0-9-]+)\s+\{([^}]+)\}',help_text)}
    version,defaults=source_metadata(path)
    item={'path':path,'kind':kind,'version':version,'defaults':defaults,'flags':flags,'choices':choices,'sha256':digest(path)}
    item['modes']=modes_for(item)
    return item,None


def paired_modes(entry,live):
    """Only expose optional switches accepted by both the entry and the live file."""
    if entry.get('kind','direct')=='direct':return entry['modes']
    shared=entry['flags'] & live['flags']
    choices={key:[value for value in values if value in live['choices'].get(key,[])]
        for key,values in entry['choices'].items() if key in shared}
    combined={**live,'flags':shared,'choices':choices,'defaults':{**live['defaults'],**entry['defaults']}}
    return modes_for(combined)


def capture_wrapper_command(entry,live,run_dir,stream_name,folder,mode_args):
    """Read the selected entry's own live command without starting the device."""
    calls=[];buffer=io.StringIO();old_argv=sys.argv[:];old_path=sys.path[:];old_cwd=Path.cwd()
    def capture(command,*args,**kwargs):
        if isinstance(command,str) or kwargs.get('shell'):raise ValueError('该入口使用shell命令，不能安全绑定；请直接选07文件。')
        calls.append([str(value) for value in command])
        return types.SimpleNamespace(returncode=0,stdout='',stderr='')
    try:
        os.chdir(entry['path'].parent);sys.path.insert(0,str(entry['path'].parent))
        with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
            namespace=runpy.run_path(str(entry['path']),run_name='rp_compare_probe')
            function=namespace.get('main')
            if not callable(function):raise ValueError('该入口没有main函数，可直接选择07文件运行。')
            globals_=function.__globals__;scripts=globals_.get('SCRIPTS')
            if not isinstance(scripts,dict) or 'live' not in scripts:
                raise ValueError('入口未声明SCRIPTS["live"]，无法临时绑定；可直接选择07文件。')
            scripts['live']=str(live['path'])
            original=globals_.get('subprocess')
            if original is None:raise ValueError('入口未通过subprocess调用实时文件。')
            class Proxy:
                def __getattr__(self,name):return capture if name=='run' else getattr(original,name)
            globals_['subprocess']=Proxy()
            sys.argv=[str(entry['path']),'live','--run',str(run_dir),'--stream-name',stream_name,'--output',str(folder),*mode_args]
            try:code=function()
            except SystemExit as exc:code=exc.code
            if code not in [None,0] or len(calls)!=1:
                raise ValueError('入口未生成唯一的实时命令：'+buffer.getvalue().strip()[-500:])
    finally:
        sys.argv=old_argv;sys.path[:]=old_path;os.chdir(old_cwd)
    command=calls[0]
    if len(command)<2 or Path(command[1]).resolve()!=live['path']:
        raise ValueError('入口未采用选中的实时文件；可直接选择07文件。')
    unsupported=sorted({value.split('=',1)[0][2:] for value in command[2:] if value.startswith('--')} - live['flags'])
    if unsupported:raise ValueError('此组合不兼容：入口转发了07文件不支持的参数 '+', '.join('--'+flag for flag in unsupported))
    for flag,values in live['choices'].items():
        option='--'+flag
        if option in command and command.index(option)+1<len(command):
            if command[command.index(option)+1] not in values:
                raise ValueError('此组合不兼容：07文件不支持%s=%s'%(option,command[command.index(option)+1]))
    activity=live['defaults'].get('--activity-mode')
    if '--activity-mode' in command:activity=command[command.index('--activity-mode')+1]
    if activity=='guided':raise ValueError('该入口未能切换掉需要提前选择动作的guided模式；可直接选择07文件的legacy模式。')
    return command


def wrapper_bootstrap(entry,live,run_dir,stream_name,folder,mode_args):
    bootstrap=folder/'selected_entry_bootstrap.py'
    bootstrap.write_text('''import json,runpy,sys
config=json.loads(sys.argv[1])
sys.path.insert(0,str(__import__('pathlib').Path(config['entry']).parent))
namespace=runpy.run_path(config['entry'],run_name='rp_selected_entry')
function=namespace['main']
function.__globals__['SCRIPTS']['live']=config['live']
sys.argv=[config['entry'],'live','--run',config['run'],'--stream-name',config['stream'],'--output',config['output'],*config['extra']]
raise SystemExit(function())
''',encoding='utf-8')
    config={'entry':str(entry['path']),'live':str(live['path']),'run':str(run_dir),'stream':stream_name,'output':str(folder),'extra':mode_args}
    return [sys.executable,str(bootstrap),json.dumps(config,ensure_ascii=False)]


def discover(directory):
    found=[];skipped=[]
    for path in sorted(directory.iterdir(),key=lambda p:p.name.casefold()):
        if not path.is_file() or path.suffix.casefold()!='.py':continue
        if not path.name.casefold().startswith(('07_live_rp_binary','run_rp_binary')):continue
        try:
            item,reason=probe(path.resolve())
            if item:found.append(item)
            else:skipped.append((path.name,reason))
        except (OSError,ValueError,subprocess.TimeoutExpired) as exc:skipped.append((path.name,str(exc)))
    return found,skipped


def choose(items,title,label):
    print('\n'+title)
    for i,item in enumerate(items,1):print('  [%d] %s'%(i,label(item)))
    while True:
        value=input('输入编号，Q返回/退出：').strip()
        if value.casefold() in ['q','quit','exit']:return None
        if value.isdigit() and 1<=int(value)<=len(items):return items[int(value)-1]
        print('请输入上面列出的有效编号。')


def read_state(directory):
    state_path=directory/'pipeline_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    if state.get('schema')!='openbci_rp_binary_pipeline_v1':raise ValueError('不是配套二分类pipeline_state.json。')
    paths=[]
    for key in ['daily_model','detector']:
        value=state.get(key)
        if not isinstance(value,str) or not value:raise ValueError('运行状态缺少'+key)
        path=Path(value).expanduser()
        if not path.is_absolute():path=directory/path
        path=path.resolve()
        if not path.is_file():raise ValueError('找不到%s：%s'%(key,path))
        paths.append(path)
    return paths


def summarize(folder):
    pack={};trigger='unknown'
    sessions=sorted(folder.rglob('session_summary.json'),key=lambda p:p.stat().st_mtime_ns)
    if sessions:
        info=json.loads(sessions[-1].read_text(encoding='utf-8-sig'))
        pack=info.get('posthoc_evaluation') or {};trigger=info.get('trigger_mode','auto')
    labels=sorted(folder.rglob('label_accuracy_summary.json'),key=lambda p:p.stat().st_mtime_ns)
    if labels:pack=json.loads(labels[-1].read_text(encoding='utf-8-sig'))
    out={key:pack.get(key) for key in ['recorded_actions','correct','wrong','manual_misses']}
    out.update(accuracy=pack.get('record_based_accuracy'),trigger_mode=trigger)
    for k,name in [(0,'stone'),(1,'paper')]:
        entry=next((row for row in pack.get('per_class',[]) if row.get('class_id')==k),{})
        out[name+'_correct']=entry.get('correct');out[name+'_total']=entry.get('record_count')
    return out


def launch(candidate,mode,model,detector,output_root,stream_name,live=None,run_dir=None):
    live=live or candidate
    stem=re.sub(r'[^\w.-]+','_',candidate['path'].stem)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    live_stem=re.sub(r'[^\w.-]+','_',live['path'].stem)
    pair_suffix='__'+live_stem if candidate.get('kind')=='wrapper' else ''
    folder=output_root/(stamp+'_'+stem+pair_suffix+'_'+mode['key'])
    before=[digest(model),digest(detector)]
    if candidate.get('kind')=='wrapper':
        if run_dir is None:raise ValueError('入口运行需要--run目录。')
        command=capture_wrapper_command(candidate,live,run_dir,stream_name,folder,mode['args'])
        for flag,expected in [('--model',model),('--detector',detector)]:
            if flag not in command or Path(command[command.index(flag)+1]).resolve()!=expected.resolve():
                raise ValueError('该入口未使用本次共用的模型/门控路径，无法作为同一模型对照。')
    else:
        command=[sys.executable,str(candidate['path']),'--model',str(model),'--detector',str(detector),
            '--stream-name',stream_name,'--output',str(folder),*mode['args']]
    folder.mkdir(parents=True,exist_ok=False)
    manifest={'started_at':datetime.now().astimezone().isoformat(),'script':str(candidate['path']),
        'version':candidate['version'],'script_sha256':candidate['sha256'],'mode':mode['key'],
        'model':str(model),'model_sha256':before[0],'detector':str(detector),'detector_sha256':before[1],
        'command':command,'result_dir':str(folder)}
    manifest.update(entry_kind=candidate.get('kind','direct'),live_script=str(live['path']),live_version=live['version'],live_sha256=live['sha256'])
    manifest_path=folder/'launch_manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n入口/启动文件：'+candidate['path'].name+'\n实际实时代码：'+live['path'].name+'\n模式：'+mode['label']+'\n结果：'+str(folder),flush=True)
    print('测试完停止并保存，再关闭识别窗口；之后会回到选择菜单。',flush=True)
    execution=wrapper_bootstrap(candidate,live,run_dir,stream_name,folder,mode['args']) if candidate.get('kind')=='wrapper' else command
    exit_code=subprocess.run(execution,cwd=str(candidate['path'].parent)).returncode
    unchanged=before==[digest(model),digest(detector)]
    manifest.update(exit_code=exit_code,inputs_unchanged=unchanged)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    row={key:manifest.get(key) for key in SUMMARY_FIELDS}
    try:row.update(summarize(folder))
    except (OSError,ValueError,TypeError) as exc:print('统计暂未读取成功：'+str(exc))
    with (folder/'comparison_row.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=SUMMARY_FIELDS);writer.writeheader();writer.writerow(row)
    target=output_root/'comparison_summary.csv';existing=target.exists() and target.stat().st_size>0
    if existing:
        try:
            with target.open(encoding='utf-8-sig',newline='') as stream:old_fields=next(csv.reader(stream),[])
        except PermissionError:old_fields=[]
        if old_fields!=SUMMARY_FIELDS:
            target=output_root/'comparison_summary_v2.csv';existing=target.exists() and target.stat().st_size>0
    try:
        with target.open('a',encoding='utf-8' if existing else 'utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=SUMMARY_FIELDS)
            if not existing:writer.writeheader()
            writer.writerow(row)
    except PermissionError:
        print('总汇总文件被占用，本次统计已保存在：'+str(folder/'comparison_row.csv'))
    value=row.get('accuracy')
    print('本次已标注正确率：'+('无已标注数据' if value is None else '%.2f%%'%(100*value)))
    print('对照汇总：'+str(target))
    if not unchanged:print('注意：模型或门控文件在测试中发生改变，本条不属于同一模型对照。')
    return exit_code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--code-dir',nargs='+',default=[str(Path(__file__).resolve().parents[1]/"binary")],help='可填写一个或多个代码目录')
    parser.add_argument('--run',default=DEFAULT_RUN,help='各版本共用的pipeline_state.json所在目录')
    parser.add_argument('--output',help='对照输出父目录，默认脚本目录旁RP_BINARY_VERSION_COMPARE')
    parser.add_argument('--stream-name',default='rps_raw')
    parser.add_argument('--list',action='store_true',help='仅列文件和可用模式，不连接设备')
    parser.add_argument('--script',help='直接指定run入口或07实时文件名')
    parser.add_argument('--live-script',help='选择run入口时指定本次实际调用的07文件')
    parser.add_argument('--mode',default='default',choices=['default','legacy','robust','rest','rest4','responsive','manual'])
    args=parser.parse_args(argv)
    try:
        directories=[Path(value).expanduser().resolve() for value in args.code_dir]
        for directory in directories:
            if not directory.is_dir():raise ValueError('代码目录不存在：'+str(directory))
        directory=directories[0]
        def resolve_script(value):
            path=Path(value).expanduser()
            if path.is_absolute():return path.resolve()
            matches=list(dict.fromkeys((folder/path).resolve() for folder in directories if (folder/path).is_file()))
            if len(matches)>1:raise ValueError('多个目录存在同名文件，请提供完整路径：'+value)
            if not matches:raise ValueError('找不到脚本：'+value)
            return matches[0]
        discovered=[];skipped=[];known=set()
        for folder in directories:
            items,ignored=discover(folder);skipped.extend(ignored)
            for item in items:
                if item['path'] not in known:discovered.append(item);known.add(item['path'])
        if args.script:
            selected=resolve_script(args.script)
            item=next((candidate for candidate in discovered if candidate['path']==selected),None)
            if item is None:item,reason=probe(selected)
            if not item:raise ValueError(reason)
            found=[item]
        else:found=discovered
        for name,reason in skipped:print('[跳过] %s：%s'%(name,reason))
        if not found:raise ValueError('未找到run_rp_binary*.py或07_live_rp_binary*.py。')
        seen={}
        for i,item in enumerate(found,1):
            same=seen.get(item['sha256']);note='；与编号%d代码完全相同'%same if same else ''
            kind='run入口' if item['kind']=='wrapper' else '直接实时'
            print('[%d] %s | %s | %s | SHA %s%s'%(i,item['path'].name,kind,item['version'],item['sha256'][:10],note))
            if len(directories)>1:print('    位置：'+str(item['path']))
            seen.setdefault(item['sha256'],i)
            if args.list:print('    可用模式：'+', '.join(mode['key'] for mode in item['modes']))
        if args.list:return 0
        if not args.run:raise ValueError("请提供--run，指向包含pipeline_state.json的二分类输出目录。")
        run_dir=Path(args.run).expanduser().resolve()
        model,detector=read_state(run_dir)
        output_root=Path(args.output).expanduser().resolve() if args.output else directory.parent/'outputs'/'VERSION_COMPARE'
        while True:
            item=found[0] if args.script else choose(found,'先选择run入口或直接实时文件：',
                lambda c:c['path'].name+' | '+('run入口' if c['kind']=='wrapper' else '直接实时')+' | '+str(c['path'].parent))
            if item is None:return 0
            live=item
            if item['kind']=='wrapper':
                if args.live_script:
                    path=resolve_script(args.live_script)
                    live=next((candidate for candidate in discovered if candidate['path']==path),None)
                    if live is None:live,reason=probe(path)
                    if not live or live['kind']!='direct':raise ValueError('--live-script须指向接受--model/--detector的实时文件。')
                elif args.script:
                    raise ValueError('指定run入口时请同时提供--live-script；或不带--script进入双选菜单。')
                else:
                    actual=[candidate for candidate in discovered if candidate['kind']=='direct']
                    if not actual:raise ValueError('未发现实际07文件，请用--code-dir包含其所在目录。')
                    live=choose(actual,'再选择这个run入口本次调用的实时文件：',lambda c:c['path'].name+' | '+c['version']+' | '+str(c['path'].parent))
                    if live is None:continue
            modes=paired_modes(item,live)
            if args.script:
                mode=next((mode for mode in modes if mode['key']==args.mode),None)
                if mode is None:raise ValueError('该组合不支持模式%s；可用：%s'%(args.mode,','.join(m['key'] for m in modes)))
            else:mode=choose(modes,'选择这个组合的识别模式：',lambda mode:mode['label'])
            if mode is None:continue
            try:code=launch(item,mode,model,detector,output_root,args.stream_name,live=live,run_dir=run_dir)
            except ValueError as exc:
                if args.script:raise
                print('[组合未启动] '+str(exc));continue
            if args.script:return code
    except KeyboardInterrupt:
        print('\n已退出通用启动器。');return 130
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        print('[启动失败] '+str(exc),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
