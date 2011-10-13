def uv2nt(u,v,x,y,x_c=0,y_c=0):
    """
    Convert orthogonal velocity components to normal and tangent ones.
    Based on x_c and y_c, with default values (0,0) convert u and v to
    normal (n) and tangent(t) velocities.
    Input:
        - u =>
        - v =>
        - x =>
        - y =>
        - x_c =>
        - y_c =>
    Output:
        - n =>
        - t =>
    """

    dx = x-x_c
    dy = y-y_c

    # Angle between x-axis and a rotated x passing throught the velocity point
    alpha = np.arctan2(dy,dx)

    n = u*np.cos(alpha)+v*np.sin(alpha)
    t = v*np.cos(alpha)-u*np.sin(alpha)

    return [n,t]

def nt2uv(n,t,x,y,x_c=0,y_c=0):
    """Convert normal and tangent velocities to Orthogonal ones

    Convert normal(n) and tangent(t) velocities to orthogonal ones,
    u and v, based on x,y of each (n,t) velocity and the circular center
    x_c and y_c
    Input:
        - n =>
        - t =>
        - x =>
        - y =>
        - x_c =>
        - y_c =>
    Output:
        - u =>
        - v =>
    """

    dx = x-x_c
    dy = y-y_c

    # Angle between x-axis and a rotated x passing throught the velocity point
    alpha = np.arctan2(dy,dx)

    u = n*np.cos(alpha) - t*np.sin(alpha);
    v = n*np.sin(alpha) + t*np.cos(alpha);

    return [u,v]

def ringcenter(x,y,u,v,n=None,method="kinetic"):
    """
    Define the ring center

    !!!Atention, this routine need a improve. It only accept 1-D data array!!!
    !!!Another suggestion is consider the possibility o the center is outside
    !!! the data domain.

    Input:
        - x =>
        - y =>
        - u =>
        - v =>
        - n =>
        - method (Criteria):
            + kinetic => highest ratio between tangent kinetic energy versus
                         normal one.
    Output:
        - x_c =>
        - y_c =>
        - E_ratio (kinetic) => For kinetic method return the energy ratio
                               matrix.
    """
    #import matplotlib.matlab

    if(method=="kinetic"):
#           if not isinstance(np.array, data): data = np.array(data)
#           Transformar em 1D:
#               data = np.ravel(data)
#               E finalmente pegar o mínimo:
#                   datamin = np.min(data)
        x_min = x[(np.argmin(x))]
        x_max = x[(np.argmax(x))]
        y_min = y[(np.argmin(y))]
        y_max = y[(np.argmax(y))]

        if(n==None):
            n = 100.

        p_x = (x_max - x_min)/n
        p_y = (y_max - y_min)/n
        x_test = np.arange(x_min,(x_max+p_x),p_x)
        y_test = np.arange((y_max+p_y),y_min,-p_y)

        I = len(x_test)
        J = len(y_test)

        E_ratio=np.zeros((J,I),np.Float)

        E_max=0
        for j in range(J):
            for i in range(I):
                [n,t] = uv2nt(u,v,x,y,x_c=x_test[i],y_c=y_test[j])
                E_ratio[j,i]=np.sum(t**2)/np.sum(n**2)

                if(E_ratio[j,i]>E_max):
                    E_max=E_ratio[j,i]
                    # print E_max
                    x_c = x_test[i]
                    y_c = y_test[j]

        # Round to 1e-3
        x_c = round(x_c,3)
        y_c = round(y_c,3)


#        output = open('teste.xyz','w')
#        for i in range(len(X_test)):
#            output.write("%10.6f %10.6f %10.6f\n" % (X_test[i],Y_test[i],E_ratio[i]))

#        output.close

        return x_c, y_c, E_ratio, x_test, y_test, E_max

    else:
        return

def adjustringcenter(t_orig, lon_orig, lat_orig, u_orig, v_orig, sections,
                     u_c=0,v _c=0, method="kinetic"):
    """Adjust ring center compensating propagation velocity

    !!! Include checks:
        - shape t=lon_orig=lat_orig=u=v
        - Must be tuples => t,lon_orig,lat_orig,u,v

    """

    print "Center velocities (start)",u_c,v_c

    section_index=sections
    ring_index = np.greater(np.sum(section_index),0)

    #Defining time series relative to first data time
    t=t_orig-min(t_orig)

    #Time duration of the cruise
    #dt=((max(t)-min(t))*24*60*60)
    dt=(max(t)-min(t))

    lon, lat, u, v = center_velocity_correction(t, lon_orig, lat_orig, u_orig,
                                                               v_orig, u_c, v_c)

    x_c_new, y_c_new, E_ratio, x_test, y_test, E_max = ringcenter(
                                                    np.compress(ring_index,lon),
                                                    np.compress(ring_index,lat),
                                                    np.compress(ring_index,u),
                                                    np.compress(ring_index,v)
                                                    )

    Dcenter=1e4
    tol=1e2
    #Improve it. The best way isn't only check if center is not moving more.
    #for i in range(20):
    while (Dcenter>tol):

#       # Estimate new center
#        [x_c_new, y_c_new, E_ratio, x_test, y_test, E_max] = ringcenter(lon,lat,u,v)
#       print "Old",x_c_new,y_c_new
#        print sections

        # Estimate the best center for each section
        x_c_s=[]
        y_c_s=[]
        for s in sections:
            #[x_c_tmp, y_c_tmp, E_ratio, x_test, y_test, E_max] = ringcenter(lon[s[0]:s[1]],lat[s[0]:s[1]],u[s[0]:s[1]],v[s[0]:s[1]],n=50)
            x_c_tmp, y_c_tmp, E_ratio, x_test, y_test, E_max = ringcenter(
                                                            np.compress(s,lon),
                                                            np.compress(s,lat),
                                                            np.compress(s,u),
                                                            np.compress(s,v),
                                                            n=50
                                                            )
            x_c_s.append(x_c_tmp)
            y_c_s.append(y_c_tmp)

#        print "Ring center",x_c_new,y_c_new
#        print "Sections centers",x_c_s,y_c_s
        #Improve it. The best way isn't only check if center is not moving more.

        # Estimate the center movement between each section samples series
        if(len(x_c_s)==2):
            dlon=x_c_s[1]-x_c_s[0]
            dlat=y_c_s[1]-y_c_s[0]
        elif(len(x_c_s)==3):
            dlon=((x_c_s[1]+x_c_s[2])-(x_c_s[0]+x_c_s[1]))/2
            dlat=((y_c_s[1]+y_c_s[2])-(y_c_s[0]+y_c_s[1]))/2
        else:
            print "uncomplete!!!"

        dx_c=dlon*60*1852
        dy_c=dlat*60*1852

#        print "velocities diff", dx_c, dx_c/dt
#        print "velocities diff", dy_c, dy_c/dt

        du_c=dx_c/dt
        dv_c=dy_c/dt
        if(du_c > .01):  du_c=.01
        if(du_c < -.01): du_c=-.01
        #if(abs(du_c)<0.01): du_c=0
        if(dv_c > .01): dv_c=.01
        if(dv_c < -.01): dv_c=-.01
        #if(abs(dv_c)<0.01): dv_c=0
        du_c = round(du_c,2)
        dv_c = round(dv_c,2)

        u_c=du_c + u_c
        v_c=dv_c + v_c

        print 'New ring velocities',u_c,v_c

        lon, lat, u, v = center_velocity_correction(t, lon_orig, lat_orig,
                                                      u_orig, v_orig, u_c, v_c )

        x_c_old = x_c_new
        y_c_old = y_c_new

        x_c_new, y_c_new, E_ratio, x_test, y_test, E_max = ringcenter(
                                                   np.compress(ring_index, lon),
                                                   np.compress(ring_index, lat),
                                                   np.compress(ring_index, u),
                                                   np.compress(ring_index, v)
                                                   )

        Dcenter=((x_c_old - x_c_new)**2+(y_c_old - y_c_new)**2)**.5
        Dcenter=Dcenter*60*1852

#        print "X-Center movement [deg]:", (x_c_old - x_c_new)
#        print "Y-Center movement [deg]:", (y_c_old - y_c_new)
#        print 'Center movement [m]',Dcenter

    x_c = round(x_c_old,3)
    y_c = round(y_c_old,3)
    u_c = round(u_c,2)
    v_c = round(v_c,2)

    print "centros",x_c_new, y_c_new
    print "velocidades",u_c, v_c


    return x_c,y_c,u_c,v_c

def center_velocity_correction(t,lon,lat,u,v,u_c,v_c):
    """Correct data due ring propagation velocity

    Input:
        -> u => Zonal velocity [m/s]
        -> v => Meridional velocity [m/s]
        -> u_c => Ring center zonal velocity [m/s]
        -> v_c => Ring center meridional velocity [m/s]

    Subtract the velocity field by the ring propagation and
      move the data position according to the ring propagation,
      so estimate what should be the sample if was done on one
      instant.
    """

    #Should I use this check here or must be applyed outside this function??
    # What's matter is the time variation and not the true time.
    #   Ex. Samples at day of year (in seconds) 350 and 351.
    delta_t=t-min(t)

    # Correct velocity field from center movement
    u_new=u-u_c
    v_new=v-v_c

    # Where was each sample when cruise start
    dx=-delta_t*u_c
    dy=-delta_t*v_c
    # Corrected latitudes due center movement
    [lon_new,lat_new]=dx2ddeg(dx,dy,lon,lat)

    return lon_new, lat_new, u_new, v_new


def dx2ddeg(dx,dy,lon,lat):
    """
    Longitude and Latitude final position due an x/y movement

    TODO: Consider an Earth projection. Maybe a good approach is convert to
    UTM and than back to Lon/Lat.
    This is not a good solution for higher latitudes
    """

    _M2NM = 0.53996e-3

    lon_new = dx * _M2NM / 60 + lon
    lat_new = dy * _M2NM / 60 + lat

    return lon_new,lat_new

def R_maximum_velocity(t,R):
    """Estimate the radius of maximum tangent velocity

    Input:
        - t => Tangent velocity
        - R => Radius of velocities data
    Output:
        - R_vmax => Radius of maximum velocity
        - precision => Precision in which R_max was defined
    """
    from fluid.common.window_mean import window_mean

    #R_max = R[np.argmax(R)]
    R_max = max(R)
    R_min = min(R)
    # !!! Improve it! A better way to define precision than fixed 2*10**3
    precision = 5e3
    #R_bin = np.arange(precision, (R_max+precision) ,precision)
    R_bin = np.arange((40e3), (R_max) ,precision)

    #t_bin = window_mean(t,R,R_bin,method='triangular',boxsize=(1.5*precision))
    t_bin = window_mean(t,R,R_bin,method='triangular',boxsize=(40e3))
    index_R_vmax = np.argmax(np.abs(t_bin))
    R_vmax = R_bin[index_R_vmax]

    return R_vmax, R_bin, t_bin