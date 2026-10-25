"""Positive segment/cone quadrature with explicit glass-grazing splits."""
import math
import numpy as np
from numba import njit
from ._prompt_numba import _segment_a_breaks


@njit(cache=True,nogil=True)
def segment_pencils(receiver,start,axis,length,density,cone,t0,dtda,origin,R,speed,
                    front_width,grade_ratio,grade_min,max_relative,phi_panel,phi_uniform,
                    max_breaks,ax,aw,gx,gw,compute_bins=True):
    displacement=receiver-start;zr=np.dot(displacement,axis);perp=displacement-zr*axis
    impact=np.sqrt(np.dot(perp,perp));cs=cone;sn=np.sqrt(max(0.,1-cs*cs))
    if sn==0:raise ValueError('segment must be above Cherenkov threshold')
    first=perp/impact if impact>0 else np.array([0.,axis[2],-axis[1]])
    if np.dot(first,first)<.01:first=np.array([-axis[2],0.,axis[0]])
    first/=np.sqrt(np.dot(first,first));second=np.cross(axis,first)
    breaks=np.empty(max_breaks)
    count=_segment_a_breaks(impact,zr,length,cs,sn,dtda if compute_bins else 0.,1/speed if compute_bins else 0.,1/speed if compute_bins else 0.,
        front_width,grade_ratio,grade_min,max_relative,breaks)
    if count>=max_breaks:raise ValueError('increase numerical max_a_breaks')
    root=zr-impact*cs/sn
    # The forward edge of the cone starts/stops intersecting the sphere here.
    cuts=np.unique(np.concatenate((breaks[:count],np.array([max(0.,min(length,root-R/sn)),max(0.,min(length,root+R/sn))]))))
    rmax=max(np.linalg.norm(displacement),np.linalg.norm(displacement-length*axis))
    if rmax<=R:raise ValueError('source lies inside glass')
    max_phi=max(phi_uniform,(int(math.ceil(2*math.asinh(math.pi*rmax/R)/phi_panel))+8)*len(gx))
    capacity=(len(cuts)-1)*len(ax)*max_phi
    radii=np.empty(capacity);angles=np.empty(capacity);clocks=np.empty(capacity);weights=np.empty(capacity)
    directions=np.empty((capacity,3));basis_x=np.empty((capacity,3));used=0
    for panel in range(len(cuts)-1):
        lo,hi=cuts[panel],cuts[panel+1]
        for ia in range(len(ax)):
            a=(lo+hi)/2+(hi-lo)/2*ax[ia];wa=(hi-lo)/2*aw[ia]
            delta=displacement-a*axis;r=np.linalg.norm(delta)
            if r<=R:raise ValueError('emission quadrature enters receiver glass')
            ray_to_om=delta/r;gamma=math.atan2(impact,zr-a);theta_min=abs(gamma-math.acos(cs))
            cap=math.asin(R/r);kphi=math.sqrt(max(0.,math.sin(gamma)*sn))
            crossing=-1.
            if kphi>0:
                cosine=(math.cos(cap)-math.cos(gamma)*cs)/(kphi*kphi)
                if -1<cosine<1:crossing=math.acos(cosine)
            uniform=kphi<1e-9 or (theta_min>.5 and crossing<0)
            if uniform:
                intervals=np.array([-math.pi,math.pi]);scale=1.
            else:
                scale=math.sqrt(theta_min*theta_min+cap*cap)/kphi
                high=math.asinh(math.pi/scale)
                edge=math.asinh(crossing/scale) if crossing>0 else 0.
                intervals=np.unique(np.array([-high,-edge,0.,edge,high]))
            for interval in range(len(intervals)-1):
                l,h=intervals[interval],intervals[interval+1]
                pieces=phi_uniform if uniform else max(1,int(math.ceil((h-l)/phi_panel)))
                nodes=1 if uniform else len(gx)
                for piece in range(pieces):
                    left=l+(h-l)*piece/pieces;right=l+(h-l)*(piece+1)/pieces
                    for node in range(nodes):
                        v=(left+right)/2 if uniform else (left+right)/2+(right-left)/2*gx[node]
                        phi=v if uniform else scale*math.sinh(v)
                        wp=1./phi_uniform if uniform else scale*math.cosh(v)*(right-left)/2*gw[node]/(2*math.pi)
                        direction=cs*axis+sn*(math.cos(phi)*first+math.sin(phi)*second)
                        cosine=min(1.,max(-1.,np.dot(direction,ray_to_om)))
                        x=ray_to_om-cosine*direction;sine=np.linalg.norm(x)
                        if sine>1e-300:x/=sine
                        else:
                            e=np.array([1.,0.,0.]) if abs(direction[0])<.9 else np.array([0.,1.,0.])
                            x=e-np.dot(e,direction)*direction;x/=np.linalg.norm(x)
                        if used>=capacity:raise ValueError('segment angular storage estimate exhausted')
                        radii[used]=r;angles[used]=math.atan2(sine,cosine)
                        clocks[used]=t0+a*dtda-origin if compute_bins else 0.;weights[used]=density*wa*wp
                        directions[used]=direction;basis_x[used]=x;used+=1
    return radii[:used],angles[:used],clocks[:used],weights[:used],directions[:used],basis_x[:used]
