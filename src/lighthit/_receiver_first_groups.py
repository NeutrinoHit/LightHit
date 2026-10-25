"""Non-singular source/acceptance groups for finite first-entry Q1 kernels.

The point-receiver Gaussian mean of 1/theta is not a source weight here.
A positive tangent-plane Gaussian cubature instead samples unresolved source
width near the receiver cone; the existing source compression is retained.
"""
import math
import numpy as np
from numba import njit
from ._prompt_numba import _prepare_pencil_shape,_geometry_bit_hash
from ._receiver_radial_rule import radial_rule
from ._prompt_numba import bit_lut_theta_index,scalar_theta_index


def share_pencil_geometry(pencils):
    """Index only repeated, bitwise equal emission positions and clocks."""
    if '_first_entry_geometry_ids' in pencils:return
    from ._prompt_numba import _exact_pencil_geometry_ids
    ids,representatives=_exact_pencil_geometry_ids(pencils['position'],pencils['time'])
    repeated=np.bincount(ids,minlength=len(representatives))>1
    rows=np.full(len(representatives),-1,np.int64)
    rows[repeated]=np.arange(int(repeated.sum()))
    pencils['_first_entry_geometry_ids']=rows[ids]
    pencils['_first_entry_geometry_representatives']=representatives[repeated]


@njit(cache=True,inline='always')
def _theta_cell(theta,minimum,step,thresholds,lut,first):
    if thresholds is None:return scalar_theta_index(theta,minimum,step)
    return bit_lut_theta_index(theta,minimum,step,thresholds,lut,first)


@njit(cache=True,inline='always')
def _weights(c1,c3,polynomial):
    x=c3;z=c1;y2=max(0.,1-x*x-z*z)
    a0,a1,a2,a3=polynomial
    return np.array([a0+a2*y2,a1*x+3*a3*x*y2,a1*z+3*a3*z*y2,
        a2*(x*x-y2),2*a2*x*z,a2*(z*z-y2),a3*(x*x*x-3*x*y2),
        3*a3*z*(x*x-y2),3*a3*x*(z*z-y2),a3*(z*z*z-3*z*y2)])


@njit(cache=True,inline='always')
def _slot6(slots,keys,a,b,c,d,e,f):
    # Full keys are compared after hashing; no bounded packing or aliasing.
    code=_geometry_bit_hash(np.uint64(a)^np.uint64(b)*np.uint64(0x9e3779b97f4a7c15)
        ^np.uint64(c)*np.uint64(0xbf58476d1ce4e5b9)^np.uint64(d)*np.uint64(0x94d049bb133111eb)
        ^np.uint64(e)*np.uint64(0xd6e8feb86659fd93)^np.uint64(f)*np.uint64(0xa0761d6478bd642f))
    i=np.int64(code&np.uint64(len(slots)-1))
    while slots[i]>=0:
        j=slots[i]
        if keys[j,0]==a and keys[j,1]==b and keys[j,2]==c and keys[j,3]==d and keys[j,4]==e and keys[j,5]==f:break
        i=(i+1)&(len(slots)-1)
    return i


@njit(cache=True,inline='always')
def _add_narrow(row,w0,w2,x,z,polynomial):
    a0,a1,a2,a3=polynomial;y2=max(0.,1-x*x-z*z)
    v0=a0+a2*y2;v1=a1*x+3*a3*x*y2;v2=a1*z+3*a3*z*y2
    v3=a2*(x*x-y2);v4=2*a2*x*z;v5=a2*(z*z-y2)
    v6=a3*(x*x*x-3*x*y2);v7=3*a3*z*(x*x-y2)
    v8=3*a3*x*(z*z-y2);v9=a3*(z*z*z-3*z*y2)
    for off,w in ((0,w0),(20,w2)):
        row[off]+=w*v0;row[off+1]+=w*v1;row[off+2]+=w*v2
        row[off+3]+=w*v3;row[off+4]+=w*v4;row[off+5]+=w*v5
        row[off+6]+=w*v6;row[off+7]+=w*v7;row[off+8]+=w*v8;row[off+9]+=w*v9



_EXPONENTS=np.array([(i,j,n-i-j) for n in range(4) for i in range(n,-1,-1)
                     for j in range(n-i,-1,-1)],np.int64)
@njit(cache=True)
def _rotate_weights(raw,rotation,polynomial):
    """Cubic look tensors -> ten finite-kernel moments in a sampled frame."""
    result=np.zeros((2,10));a0,a1,a2,a3=polynomial
    x=rotation[:,0];y=rotation[:,1];z=rotation[:,2]
    for field in range(2):
        r=raw[field];vx=x[0]*r[1]+x[1]*r[2]+x[2]*r[3]
        vz=z[0]*r[1]+z[1]*r[2]+z[2]*r[3]
        yy=_second(r,y,y);xx=_second(r,x,x);xz=_second(r,x,z);zz=_second(r,z,z)
        xyy=0.;zyy=0.;xxx=0.;xxz=0.;xzz=0.;zzz=0.
        if a3!=0.:
            xyy=_third(r,x,y,y);zyy=_third(r,z,y,y);xxx=_third(r,x,x,x)
            xxz=_third(r,x,x,z);xzz=_third(r,x,z,z);zzz=_third(r,z,z,z)
        result[field,0]=a0*r[0]+a2*yy
        result[field,1]=a1*vx+3*a3*xyy;result[field,2]=a1*vz+3*a3*zyy
        result[field,3]=a2*(xx-yy);result[field,4]=2*a2*xz;result[field,5]=a2*(zz-yy)
        result[field,6]=a3*(xxx-3*xyy);result[field,7]=3*a3*(xxz-zyy)
        result[field,8]=3*a3*(xzz-xyy);result[field,9]=a3*(zzz-3*zyy)
    return result


@njit(cache=True,inline='always')
def _second(r,a,b):
    return (r[4]*a[0]*b[0]+r[7]*a[1]*b[1]+r[9]*a[2]*b[2]
        +r[5]*(a[0]*b[1]+a[1]*b[0])+r[6]*(a[0]*b[2]+a[2]*b[0])
        +r[8]*(a[1]*b[2]+a[2]*b[1]))


@njit(cache=True,inline='always')
def _third(r,a,b,c):
    # Contract only the ten independent components of a symmetric tensor.
    return (r[10]*a[0]*b[0]*c[0]+r[16]*a[1]*b[1]*c[1]+r[19]*a[2]*b[2]*c[2]
        +r[11]*(a[0]*b[0]*c[1]+a[0]*b[1]*c[0]+a[1]*b[0]*c[0])
        +r[12]*(a[0]*b[0]*c[2]+a[0]*b[2]*c[0]+a[2]*b[0]*c[0])
        +r[13]*(a[0]*b[1]*c[1]+a[1]*b[0]*c[1]+a[1]*b[1]*c[0])
        +r[15]*(a[0]*b[2]*c[2]+a[2]*b[0]*c[2]+a[2]*b[2]*c[0])
        +r[17]*(a[1]*b[1]*c[2]+a[1]*b[2]*c[1]+a[2]*b[1]*c[1])
        +r[18]*(a[1]*b[2]*c[2]+a[2]*b[1]*c[2]+a[2]*b[2]*c[1])
        +r[14]*(a[0]*b[1]*c[2]+a[0]*b[2]*c[1]+a[1]*b[0]*c[2]
                  +a[1]*b[2]*c[0]+a[2]*b[0]*c[1]+a[2]*b[1]*c[0]))


@njit(cache=True)
def _regroup(radii,angles,clocks,spreads,moments,masses,R,radial_log_step,
             clock_step,theta_min,theta_ratio,hit_nodes,miss_nodes,speed_ref,coarse=False,
             theta_thresholds=None,theta_lut=None,theta_first=0):
    """Merge sampled groups back onto the same geometry/clock cells.

    Cubature increases angular integration resolution, not the number of
    distinct interpolated receiver kernels. Both field moments and the
    mixture's first two source-clock moments are retained.
    """
    slots=np.full(8,-1,np.int64);keys=np.empty((4,6),np.int64)
    values=np.zeros((4,26));used=0;log_theta=math.log1p(theta_ratio)
    last_r=0.;cap=0.;radial_key=0;flight=0.
    for p in range(len(radii)):
        r=radii[p];theta=angles[p]
        if p==0 or r!=last_r:
            cap=math.asin(R/r);radial_key=int(math.floor(math.log(r)/radial_log_step))
            flight=r/speed_ref;last_r=r
        hit=theta<=cap
        coordinate=r*math.sin(theta)/R if hit else theta
        cell=0
        theta_cell=_theta_cell(theta,theta_min,log_theta,theta_thresholds,theta_lut,theta_first)
        tau=clocks[p]+flight
        k0=radial_key;k1=int(math.floor(tau/clock_step))
        k2=int(hit);k3=cell;k4=theta_cell
        if 2*(used+1)>len(slots):
            fresh=np.full(len(slots)*2,-1,np.int64)
            larger_keys=np.empty((len(keys)*2,6),np.int64);larger_keys[:used]=keys[:used]
            larger=np.zeros((len(values)*2,26));larger[:used]=values[:used]
            for old in range(used):
                fresh[_slot6(fresh,larger_keys,larger_keys[old,0],larger_keys[old,1],larger_keys[old,2],
                    larger_keys[old,3],larger_keys[old,4],0)]=old
            slots=fresh;keys=larger_keys;values=larger
        slot=_slot6(slots,keys,k0,k1,k2,k3,k4,0)
        if slots[slot]<0:
            slots[slot]=used;keys[used,0]=k0;keys[used,1]=k1;keys[used,2]=k2
            keys[used,3]=k3;keys[used,4]=k4;keys[used,5]=0;used+=1
        row=values[slots[slot]]
        for field in range(2):
            for m in range(10):row[field*10+m]+=moments[p,field,m]
        mass=masses[p];new_mass=row[20]+mass;difference=tau-row[24]
        row[24]+=mass/new_mass*difference
        row[25]+=mass*(difference*(tau-row[24])+spreads[p]*spreads[p]/12)
        row[20]=new_mass;row[21]+=mass*r;row[22]+=mass*coordinate;row[23]+=mass*clocks[p]
    out_r=np.empty(used);out_theta=np.empty(used);out_clock=np.empty(used);out_spread=np.empty(used)
    out_m=np.empty((used,2,10))
    for j in range(used):
        row=values[j];mass=row[20];r=row[21]/mass;x=row[22]/mass;cap=math.asin(R/r)
        if keys[j,2]:theta=min(cap,math.asin(min(1.,max(0.,x*R/r))))
        else:
            theta=x
            if theta<=cap:theta=np.nextafter(cap,np.inf)
        out_r[j]=r;out_theta[j]=theta;out_clock[j]=row[23]/mass
        out_spread[j]=math.sqrt(12*max(0.,row[25]/mass))
        out_m[j,0]=row[:10];out_m[j,1]=row[10:20]
    return out_r,out_theta,out_clock,out_spread,out_m


@njit(cache=True,nogil=True)
def prepare(receiver,look,origin,position,direction,w0,w2,clocks,major,var_major,
            var_minor,prepared,polynomial,R,radial_log_step,clock_step,theta_min,
            theta_ratio,hit_nodes,miss_nodes,speed_ref,gauss_x,gauss_w,covariance_tolerance,selected=None,radial_only=False,
            hierarchy_parent=None,next_node=None,hierarchy_smoothing=6.,
            theta_thresholds=None,theta_lut=None,theta_first=0,geometry_ids=None,geometry_count=0):
    """Group first, then integrate each group's unresolved Gaussian width.

    Each group retains both fields' full cubic look tensors, the mean
    tangent covariance and source-clock variance. Gaussian mixtures within
    one geometry group use their mean covariance; refining the grouping
    checks that additional source approximation. No 1/theta factor is used.
    """
    capacity=8
    slots=np.full(capacity,-1,np.int64);keys=np.empty((capacity//2,6),np.int64)
    values=np.zeros((capacity//2,49));used=0;shape_samples=0
    log_theta=math.log1p(theta_ratio);powers=np.empty((3,4))
    raw_count=10 if polynomial[3]==0. else 20
    geometry=np.empty((geometry_count,10));geometry_keys=np.empty((geometry_count,2),np.int64)
    geometry_ready=np.zeros(geometry_count,np.bool_)
    count=len(w0) if selected is None else len(selected)
    cursor=0
    while cursor<count:
        index=cursor;cursor+=1
        p=index if selected is None else selected[index]
        if w0[p]==0. and w2[p]==0.:continue
        gid=-1 if geometry_ids is None else geometry_ids[p]
        if gid>=0 and geometry_ready[gid]:
            dx=geometry[gid,0];dy=geometry[gid,1];dz=geometry[gid,2];r=geometry[gid,3]
            hx=geometry[gid,4];hy=geometry[gid,5];hz=geometry[gid,6]
            cap=geometry[gid,7];t_emit=geometry[gid,8];tau=geometry[gid,9]
            k0=geometry_keys[gid,0];k1=geometry_keys[gid,1]
        else:
            dx=receiver[0]-position[p,0];dy=receiver[1]-position[p,1];dz=receiver[2]-position[p,2]
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            if r<=R:raise ValueError('compressed source lies inside the glass')
            hx=dx/r;hy=dy/r;hz=dz/r;cap=math.asin(R/r)
            t_emit=clocks[p]-origin;tau=t_emit+r/speed_ref
            k0=int(math.floor(math.log(r)/radial_log_step));k1=int(math.floor(tau/clock_step))
            if gid>=0:
                geometry[gid,0]=dx;geometry[gid,1]=dy;geometry[gid,2]=dz;geometry[gid,3]=r
                geometry[gid,4]=hx;geometry[gid,5]=hy;geometry[gid,6]=hz
                geometry[gid,7]=cap;geometry[gid,8]=t_emit;geometry[gid,9]=tau
                geometry_keys[gid,0]=k0;geometry_keys[gid,1]=k1;geometry_ready[gid]=True
        sx=direction[p,0];sy=direction[p,1];sz=direction[p,2]
        cosine=min(1.,max(-1.,hx*sx+hy*sy+hz*sz))
        if hierarchy_parent is not None and hierarchy_parent[p]==-2:
            parent_cosine=(dx*sx+dy*sy+dz*sz)/r
            parent_theta=math.acos(min(1.,max(-1.,parent_cosine)))
            extent=var_major[p]+prepared[p,6]/(r*r)
            gap=max(0.,parent_theta-cap)
            if parent_theta<.5*math.pi and gap*gap<hierarchy_smoothing*hierarchy_smoothing*extent:continue
            cursor=next_node[p]
        px=hx-cosine*sx;py=hy-cosine*sy;pz=hz-cosine*sz
        sine=math.sqrt(px*px+py*py+pz*pz);theta=math.atan2(sine,cosine)
        ax=major[p,0];ay=major[p,1];az=major[p,2]
        bx=prepared[p,0];by=prepared[p,1];bz=prepared[p,2]
        if sine>1e-300:
            nx=px/sine;ny=py/sine;nz=pz/sine
        else:nx=ax;ny=ay;nz=az
        lx=look[0]*nx+look[1]*ny+look[2]*nz
        lz=look[0]*sx+look[1]*sy+look[2]*sz
        m11=max(0.,var_major[p]+prepared[p,3]/(r*r))
        m22=max(0.,var_minor[p]+prepared[p,4]/(r*r));m12=prepared[p,5]/(r*r)
        cxx=0.;cyy=0.;cxy=0.;wide=False
        threshold=covariance_tolerance*(theta*theta+(R/r)**2)
        if 1.5*(m11+m22)>threshold:
            ca=ax*nx+ay*ny+az*nz;sa=bx*nx+by*ny+bz*nz
            cxx=ca*ca*m11+2*ca*sa*m12+sa*sa*m22
            cyy=sa*sa*m11-2*ca*sa*m12+ca*ca*m22
            cxy=ca*sa*(m22-m11)+(ca*ca-sa*sa)*m12
            wide=max(cxx,cyy)+abs(cxy)>threshold
        if radial_only and theta>math.pi/2:wide=False
        hit=theta<=cap
        # Source merging follows the established relative-angle grid. It
        # must not be coupled to the storage cells of a particular atlas.
        coordinate=r*sine/R if hit else theta
        atlas_cell=0
        theta_cell=_theta_cell(theta,theta_min,log_theta,theta_thresholds,theta_lut,theta_first)
        k2=int(hit);k3=atlas_cell;k4=theta_cell;k5=int(wide)
        if 2*(used+1)>len(slots):
            fresh=np.full(len(slots)*2,-1,np.int64)
            larger_keys=np.empty((len(keys)*2,6),np.int64);larger_keys[:used]=keys[:used]
            larger=np.zeros((len(values)*2,49));larger[:used]=values[:used]
            for old in range(used):
                fresh[_slot6(fresh,larger_keys,larger_keys[old,0],larger_keys[old,1],larger_keys[old,2],
                    larger_keys[old,3],larger_keys[old,4],larger_keys[old,5])]=old
            slots=fresh;keys=larger_keys;values=larger
        slot=_slot6(slots,keys,k0,k1,k2,k3,k4,k5)
        if slots[slot]<0:
            slots[slot]=used;keys[used,0]=k0;keys[used,1]=k1;keys[used,2]=k2
            keys[used,3]=k3;keys[used,4]=k4;keys[used,5]=k5;used+=1
        row=values[slots[slot]]
        if wide and not radial_only:
            yx=sy*nz-sz*ny;yy=sz*nx-sx*nz;yz=sx*ny-sy*nx
            ly=look[0]*yx+look[1]*yy+look[2]*yz
            for a in range(3):powers[a,0]=1.
            for k in range(1,4):
                powers[0,k]=powers[0,k-1]*lx;powers[1,k]=powers[1,k-1]*ly;powers[2,k]=powers[2,k-1]*lz
            for m in range(raw_count):
                e=_EXPONENTS[m];factor=powers[0,e[0]]*powers[1,e[1]]*powers[2,e[2]]
                row[m]+=w0[p]*factor;row[20+m]+=w2[p]*factor
        else:
            _add_narrow(row,w0[p],w2[p],lx,lz,polynomial)
        mass=abs(w0[p]) if w0[p]!=0. else abs(w2[p]);new_mass=row[40]+mass
        difference=tau-row[44];row[44]+=mass/new_mass*difference
        row[45]+=mass*difference*(tau-row[44]);row[40]=new_mass
        row[41]+=mass*r;row[42]+=mass*coordinate;row[43]+=mass*t_emit
        row[46]+=mass*cxx;row[47]+=mass*cxy;row[48]+=mass*cyy
    maximum=used*(2 if radial_only else len(gauss_x)**2)
    radii=np.empty(maximum);angles=np.empty(maximum);time=np.empty(maximum);spread=np.empty(maximum)
    moments=np.empty((maximum,2,10));masses=np.empty(maximum);emitted=0
    for j in range(used):
        row=values[j];mass=row[40];r=row[41]/mass;coordinate=row[42]/mass;cap=math.asin(R/r)
        group_time=row[43]/mass;group_spread=math.sqrt(12*max(0.,row[45]/mass))
        if keys[j,2]:theta=min(cap,math.asin(min(1.,max(0.,coordinate*R/r))))
        else:
            theta=coordinate
            if theta<=cap:theta=np.nextafter(cap,np.inf)
        cxx=max(0.,row[46]/mass);cxy=row[47]/mass;cyy=max(0.,row[48]/mass)
        cubature=keys[j,5]!=0
        if radial_only and cubature:
            half=(cxx+cyy)/2;disc=math.sqrt(max(0.,(cxx-cyy)**2/4+cxy*cxy))
            angle=.5*math.atan2(2*cxy,cxx-cyy)
            p0,t0,p1,t1=radial_rule(math.cos(angle)*theta,-math.sin(angle)*theta,
                max(0.,half+disc),max(0.,half-disc),cap,gauss_x,gauss_w)
            for probability,sample_theta in ((p0,t0),(p1,t1)):
                if probability==0:continue
                radii[emitted]=r;angles[emitted]=sample_theta;time[emitted]=group_time
                spread[emitted]=group_spread;masses[emitted]=probability*mass
                moments[emitted,0]=probability*row[:10];moments[emitted,1]=probability*row[20:30]
                emitted+=1;shape_samples+=1
            continue
        l11=math.sqrt(cxx);l21=cxy/l11 if l11>0 else 0.;l22=math.sqrt(max(0.,cyy-l21*l21))
        count=len(gauss_x) if cubature else 1
        raw=np.empty((2,20));raw[0]=row[:20];raw[1]=row[20:40]
        h=np.array([math.sin(theta),0.,math.cos(theta)])
        for first in range(count):
            for second in range(count):
                weight=gauss_w[first]*gauss_w[second] if cubature else 1.
                rotation=np.eye(3);sample_theta=theta
                if cubature:
                    z=np.array([l11*gauss_x[first],l21*gauss_x[first]+l22*gauss_x[second],1.])
                    z/=math.sqrt(np.dot(z,z));cosine=min(1.,max(-1.,np.dot(h,z)))
                    x=h-cosine*z;sine=math.sqrt(np.dot(x,x));sample_theta=math.atan2(sine,cosine)
                    if sine>1e-300:x/=sine
                    else:
                        x=np.array([1.,0.,0.])-z[0]*z
                        x/=math.sqrt(np.dot(x,x))
                    y=np.cross(z,x)
                    rotation[:,0]=x;rotation[:,1]=y;rotation[:,2]=z;shape_samples+=1
                radii[emitted]=r;angles[emitted]=sample_theta;time[emitted]=group_time
                spread[emitted]=group_spread
                if cubature:moments[emitted]=weight*_rotate_weights(raw,rotation,polynomial)
                else:
                    moments[emitted,0]=row[:10];moments[emitted,1]=row[20:30]
                masses[emitted]=weight*mass;emitted+=1
    result=_regroup(radii[:emitted],angles[:emitted],time[:emitted],spread[:emitted],moments[:emitted],masses[:emitted],
        R,radial_log_step,clock_step,theta_min,theta_ratio,hit_nodes,miss_nodes,speed_ref,radial_only,
        theta_thresholds,theta_lut,theta_first)
    return result[0],result[1],result[2],result[3],result[4],shape_samples


@njit(cache=True,nogil=True)
def select_screen(receiver,position,direction,var_major,shape,parent,next_node,R,smoothing_factor):
    """Select each parent or its children, including the finite glass cone."""
    selected=np.zeros(len(parent),np.bool_);cursor=0
    while cursor<len(parent):
        p=cursor;cursor+=1
        if parent[p]==-2:
            dx=receiver[0]-position[p,0];dy=receiver[1]-position[p,1];dz=receiver[2]-position[p,2]
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            if r<=R:raise ValueError('compressed source lies inside the glass')
            cosine=(dx*direction[p,0]+dy*direction[p,1]+dz*direction[p,2])/r
            theta=math.acos(min(1.,max(-1.,cosine)))
            extent=var_major[p]+shape[p,6]/(r*r)
            gap=max(0.,theta-math.asin(R/r))
            if theta<.5*math.pi and gap*gap<smoothing_factor*smoothing_factor*extent:continue
            cursor=next_node[p]
        selected[p]=True
    return selected
