# -*- coding: utf-8 -*-
"""Independent OpenBCI LSL diagnostic; does not modify training or QC settings."""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


def analyze(x, t, arrivals, phases, duration, fs, out, notch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = np.diff(t)
    valid = np.isfinite(dt)
    positive = dt[valid & (dt > 0)]
    count = len(x)
    bins = np.arange(int(np.ceil(duration)) + 1, dtype=float)
    bins[-1] = duration
    counts = np.histogram(arrivals, bins=bins)[0]
    elapsed = np.diff(bins)
    rates = counts / elapsed
    expected = duration * fs
    report = dict(samples=count, duration_s=duration, nominal_hz=fs,
                  expected_samples_approx=expected, arrival_rate_hz=count/duration,
                  timestamp_rate_hz=float((count-1)/(t[-1]-t[0])) if count>1 and t[-1]>t[0] else None,
                  nonincreasing_count=int(np.sum(dt[valid]<=0)), nonfinite_intervals=int(np.sum(~valid)),
                  nonincreasing_fraction=float(np.sum(dt[valid]<=0)/max(1,len(dt))),
                  gap_over_12ms=int(np.sum(dt[valid]>.012)), gap_over_50ms=int(np.sum(dt[valid]>.05)),
                  gap_over_100ms=int(np.sum(dt[valid]>.1)),
                  max_gap_ms=float(np.max(positive)*1000) if len(positive) else None,
                  positive_interval_percentiles_ms=dict(zip(['p50','p95','p99'],map(float,np.percentile(positive*1000,[50,95,99])))) if len(positive) else {},
                  per_second_arrival_counts=counts.tolist(), channels=[], phase_rms=[],
                  note='Arrival counts and raw LSL timestamps cannot prove hardware packet loss. Verify GUI Sample Index separately. RMS uses sample order at nominal rate; interpret cautiously if timing is abnormal.')
    with (out/'samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['sample_row','lsl_timestamp','arrival_elapsed_s','cue_phase','CH1','CH2','CH3','CH4'])
        w.writerows((i, t[i], arrivals[i], phases[i], *x[i]) for i in range(count))
    with (out/'timestamp_anomalies.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['previous_row','current_row','delta_ms','arrival_s'])
        w.writerows((i,i+1,dt[i]*1000,arrivals[i+1]) for i in np.flatnonzero((dt<=0)|(dt>.012)|~valid))
    for c in range(4):
        v=x[:,c]; z=v[np.isfinite(v)]
        report['channels'].append(dict(channel=c+1, nonfinite=int(len(v)-len(z)), mean=float(np.mean(z)) if len(z) else None, sd=float(np.std(z)) if len(z) else None, peak_to_peak=float(np.ptp(z)) if len(z) else None, flat=bool(len(z)>0 and np.ptp(z)<1e-7), abs_over_100000_fraction=float(np.mean(np.abs(z)>100000)) if len(z) else None))
    # Use only contiguous finite blocks, never interpolate or sort timestamps.
    y=np.full_like(x,np.nan); good=np.all(np.isfinite(x),axis=1)
    # Split on major timestamp gaps or arrival pauses; smaller jitter remains flagged.
    cut=np.flatnonzero((np.diff(arrivals)>.1)|(dt>.1)|~valid)+1
    edges=np.unique(np.r_[0,cut,np.flatnonzero(np.diff(good.astype(int)))+1,count])
    sos=butter(4,[20,100],btype='bandpass',fs=fs,output='sos')
    for a,b in zip(edges[:-1],edges[1:]):
        if b-a<fs or not good[a:b].all(): continue
        v=sosfiltfilt(sos,x[a:b],axis=0)
        if 0<notch<fs/2:
            nb,na=iirnotch(notch,30,fs=fs); v=filtfilt(nb,na,v,axis=0)
        y[a:b]=v
    rms=[]; rt=[]; rp=[]; width=int(fs*.2)
    for a in range(0,count-width+1,width):
        b=a+width
        if not np.isfinite(y[a:b]).all() or phases[a]!=phases[b-1] or np.any((dt[a:b-1]>.1)|(np.diff(arrivals[a:b])>.1)): continue
        rms.append(np.sqrt(np.mean(y[a:b]**2,axis=0))); rt.append(float(np.mean(arrivals[a:b]))); rp.append(phases[a])
    rms=np.asarray(rms).reshape(-1,4); rp=np.asarray(rp)
    for phase in ['REST','ROCK','PAPER']:
        v=rms[rp==phase]
        report['phase_rms'].append(dict(phase=phase, windows=len(v), median_rms=np.median(v,axis=0).tolist() if len(v) else None, p90_rms=np.percentile(v,90,axis=0).tolist() if len(v) else None))
    report['timing_status']='WARNING' if (abs(count/duration-fs)>fs*.03 or report['nonincreasing_count'] or report['gap_over_12ms'] or report['nonfinite_intervals']) else 'NO_TIMING_WARNING_OBSERVED'
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    fig,ax=plt.subplots(4,1,figsize=(12,8),sharex=True)
    for c in range(4):
        ax[c].plot(arrivals,x[:,c],lw=.5); ax[c].set_ylabel(f'CH{c+1} raw')
        for v in [20,40]: ax[c].axvline(v,ls='--',color='gray')
    ax[-1].set_xlabel('Arrival elapsed (s)'); fig.tight_layout(); fig.savefig(out/'raw_waveforms.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(11,7)); ax[0].plot(arrivals[1:],dt*1000,lw=.5); ax[0].axhline(4,ls='--'); ax[0].set_ylabel('LSL interval (ms)'); ax[0].set_xlabel('Arrival elapsed (s)'); ax[1].hist(dt[valid]*1000,bins=100); ax[1].set_xlabel('LSL interval (ms)'); ax[1].set_ylabel('Count'); fig.tight_layout(); fig.savefig(out/'timestamp_intervals.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4)); ax.bar(bins[:-1],rates,width=.9,align='edge'); ax.axhline(fs,ls='--'); ax.set(xlabel='Elapsed (s)',ylabel='Received samples / second'); fig.tight_layout(); fig.savefig(out/'arrival_rate.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));
    for c in range(4): ax.plot(rt,rms[:,c],label=f'CH{c+1}')
    for v in [20,40]: ax.axvline(v,ls='--',color='gray')
    ax.set(xlabel='Arrival elapsed (s)',ylabel='Filtered RMS (stream units)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'rms_curves.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));
    for j,p in enumerate(report['phase_rms']):
        if p['median_rms'] is not None: ax.bar(np.arange(4)+j*.25,p['median_rms'],width=.24,label=p['phase'])
    ax.set_xticks(np.arange(4)+.25,['CH1','CH2','CH3','CH4']); ax.set_ylabel('Median RMS (stream units)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'phase_rms.png',dpi=140); plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser(description='60-second OpenBCI raw LSL diagnostic'); p.add_argument('--stream-name',default='rps_raw'); p.add_argument('--stream-type',default='EMG'); p.add_argument('--notch',type=float,default=50); p.add_argument('--out',default='outputs/lsl_diagnosis'); p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    out=Path(a.out)/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); out.mkdir(parents=True)
    fs=250.; duration=60.
    if a.self_test:
        rng=np.random.default_rng(42); arrivals=np.arange(15000)/fs; t=100+arrivals.copy(); t[100]=t[99]; t[500]+=.016; x=rng.normal(size=(15000,4)); phases=np.where(arrivals<20,'REST',np.where(arrivals<40,'ROCK','PAPER')); x[phases=='ROCK',:2]*=4; x[phases=='PAPER',2:]*=4
    else:
        from pylsl import resolve_byprop, StreamInlet
        streams=[s for s in resolve_byprop('name',a.stream_name,timeout=10) if s.type()==a.stream_type]
        if len(streams)!=1: raise RuntimeError(f'Expected exactly one matching LSL stream; found {len(streams)}')
        info=streams[0]; print(f'LSL: {info.name()} / {info.type()} / {info.channel_count()} channels / {info.nominal_srate()} Hz')
        if info.channel_count()<4 or abs(info.nominal_srate()-fs)>1: raise RuntimeError('Need >=4 channels and nominal 250 Hz')
        inlet=StreamInlet(info,max_buflen=10,recover=False,processing_flags=0)
        inlet.open_stream(timeout=10)
        input('GUI recording ON. Press Enter to start 60 seconds...'); inlet.flush()
        xs=[]; ts=[]; ars=[]; ps=[]; last=None; start=time.monotonic(); print('0-20s REST: relax',flush=True)
        try:
            while (e:=time.monotonic()-start)<duration:
                phase='REST' if e<20 else 'ROCK' if e<40 else 'PAPER'
                cue=phase if phase=='REST' else phase+(' HOLD' if int(e%20)%4<2 else ' RELAX')
                if cue!=last: print(f'{e:5.1f}s: {cue}',flush=True); last=cue
                chunk,stamp=inlet.pull_chunk(timeout=.05,max_samples=256); at=time.monotonic()-start
                if chunk and at<duration:
                    xs.extend(np.asarray(chunk)[:,:4]); ts.extend(stamp); ars.extend([at]*len(chunk)); ps.extend(['REST' if at<20 else 'ROCK' if at<40 else 'PAPER']*len(chunk))
        finally: inlet.close_stream()
        x=np.asarray(xs,float).reshape(-1,4); t=np.asarray(ts,float); arrivals=np.asarray(ars,float); phases=np.asarray(ps)
        if len(x)<2: raise RuntimeError('Too few samples received; check stream')
    r=analyze(x,t,arrivals,phases,duration,fs,out,a.notch)
    print(json.dumps(r,ensure_ascii=False,indent=2)); print('Results:',out.resolve()); print('Synthetic test only' if a.self_test else 'Check GUI Sample Index separately before inferring packet loss.')

if __name__=='__main__': main()
