"""
Component functions of the STG nutrient budget. Individual functions are accelerated with numba just in time compiling.
"""

import numpy as np
import numba
import copy

@numba.jit()# possible options: nopython=True, parallel=True)
def d_dr(field,r):
    dfield_dr = np.zeros_like(field)
    for i in xrange(1,len(r)-1):
        dfield_dr[i] = (field[i+1] - field[i-1])/(r[i+1] - r[i-1])
        
    dfield_dr[0] = (field[1] - field[0])/(r[1] - r[0])
    dfield_dr[-1] = (field[-1] - field[-2])/(r[-1] - r[-2])
    
    return dfield_dr

def integrate_field_dr(field,r):
    int_field_dr = np.zeros_like(field)
    delta_r = r[1] - r[0]
    int_field_dr = np.cumsum(field*2.*np.pi*r*delta_r)
    
    return int_field_dr

@numba.jit()
def hor_diff(kappa_h, r, h, dPO4_dr):
    hor_diff_term = d_dr(kappa_h*r*h*dPO4_dr, r)/r
    return hor_diff_term

@numba.jit()
def hor_Ek_transport(w_ek, dPO4rr_dr, r):
    hor_Ek_trans_term = w_ek*dPO4rr_dr/(2*r)
    return hor_Ek_trans_term

@numba.jit()
def ver_Ek(PO4, w_ek):
    ver_Ek_term = PO4*w_ek
    return ver_Ek_term

@numba.jit()
def EZ_productivity(h_euph, alpha, PO4, lambda_po4):
    EZ_prod_term = h_euph*alpha*PO4/(PO4 + lambda_po4)
    return EZ_prod_term

@numba.jit()
def DOP_remineralisation(h, gamma_remin, DOP):
    DOP_remin_term = h*gamma_remin*DOP 
    return DOP_remin_term


@numba.jit()
def vert_diff_mode_to_EZ(kappa_z, PO4, PO4M, h_euph):
    vert_diff_mode_to_EZ_term = kappa_z*(PO4M - PO4)/(h_euph)
    return vert_diff_mode_to_EZ_term

@numba.jit()
def eddy_pumping_mode_to_EZ(w_rms, A_eddy, E_eddy, PO4, PO4M):
    eddy_pumping_mode_to_EZ_term = w_rms*A_eddy*E_eddy*(PO4M - PO4)
    return eddy_pumping_mode_to_EZ_term

@numba.jit()
def vert_diff_abyss_to_mode(kappa_z, PO4abyss, PO4M, h):
    vert_diff_abyss_to_mode_term = kappa_z*(PO4abyss - PO4M)/h
    return vert_diff_abyss_to_mode_term

@numba.jit()
def particulate_remineralisation(f_DOP, h_euph, h_mode, alpha, PO4, lambda_po4):
    particulate_remin_term = ((1.-f_DOP)*EZ_productivity(h_euph, alpha, PO4, lambda_po4)*
                            (1.-(((h_mode+h_euph)/h_euph)**(-0.988))))
    return particulate_remin_term

@numba.jit()
def calc_dPO4_dt(PO4, PO4M, DOP, kappa_h, kappa_z, r, h_euph, h_mode, dPO4_dr, w_ek, dPO4rr_dr,
            alpha, lambda_po4, gamma_remin, w_rms, A_eddy, E_eddy):

    PO4_time_deriv = (hor_diff(kappa_h, r, h_euph, dPO4_dr)
               - hor_Ek_transport(w_ek, dPO4rr_dr, r)
               + ver_Ek(PO4, w_ek)
               - EZ_productivity(h_euph, alpha, PO4, lambda_po4)
               + DOP_remineralisation(h_euph, gamma_remin, DOP)
               + vert_diff_mode_to_EZ(kappa_z, PO4, PO4M, h_euph)
               + eddy_pumping_mode_to_EZ(w_rms, A_eddy,
                                         E_eddy, PO4, PO4M)
                )/h_euph
    return PO4_time_deriv

@numba.jit()
def calc_dPO4M_dt(PO4, PO4M, DOPM, dPO4M_dr, dPO4Mrr_dr, kappa_h, 
                  kappa_z, r, h_euph, h_mode, w_ek, f_DOP, alpha,
                  lambda_po4, gamma_remin, PO4abyss,
                  w_rms, A_eddy, E_eddy):
        
        PO4M_time_deriv = (
                          hor_diff(kappa_h, r, h_mode, dPO4M_dr)
                        + hor_Ek_transport(w_ek, dPO4Mrr_dr, r)
                        - ver_Ek(PO4, w_ek)
                        + particulate_remineralisation(f_DOP, h_euph, h_mode,
                                                       alpha, PO4, lambda_po4)
                        + DOP_remineralisation(h_mode, gamma_remin, DOPM)
                        - vert_diff_mode_to_EZ(kappa_z, PO4, PO4M, h_euph)
                        + vert_diff_abyss_to_mode(kappa_z, PO4abyss, PO4M, h_mode)
                        - eddy_pumping_mode_to_EZ(w_rms, A_eddy,
                                                 E_eddy, PO4, PO4M)  
                        )/h_mode
        
        return PO4M_time_deriv

@numba.jit()
def calc_dDOPdt(PO4, DOP, DOPM, dDOP_dr, dDOPrr_dr, kappa_h, 
                                     kappa_z, r, 
                                    h_euph, h_mode, w_ek,
                                    f_DOP, alpha, lambda_po4,
                                    gamma_remin, PO4abyss,
                                    w_rms, A_eddy, E_eddy):
        
        DOP_time_deriv = (
                         hor_diff(kappa_h, r, h_euph, dDOP_dr)
                        - hor_Ek_transport(w_ek, dDOPrr_dr, r)
                        + ver_Ek(DOP, w_ek)
                        + f_DOP*EZ_productivity(h_euph, alpha, PO4,
                                                          lambda_po4)
                        - DOP_remineralisation(h_euph, gamma_remin, DOP)
                        + vert_diff_mode_to_EZ(kappa_z, DOP, DOPM, h_euph)
                        + eddy_pumping_mode_to_EZ(w_rms, A_eddy,
                                                 E_eddy, DOP, DOPM)
                        )/h_euph
        return DOP_time_deriv

@numba.jit()
def calc_DOPM_dt(PO4M, DOP, DOPM, dDOPM_dr, dDOPMrr_dr, kappa_h, 
                                     kappa_z, r, 
                                    h_euph, h_mode, w_ek,
                                    f_DOP, alpha, lambda_po4,
                                    gamma_remin, DOPabyss,
                                    w_rms, A_eddy, E_eddy):
        
        DOPM_time_deriv = (
                          hor_diff(kappa_h, r, h_mode, dDOPM_dr)
                        + hor_Ek_transport(w_ek, dDOPMrr_dr, r)
                        - ver_Ek(DOP, w_ek)
                        - DOP_remineralisation(h_mode, gamma_remin, DOPM)
                        - vert_diff_mode_to_EZ(kappa_z, DOP, DOPM, h_euph)
                        + vert_diff_abyss_to_mode(kappa_z, DOPabyss, DOPM, h_mode)
                        - eddy_pumping_mode_to_EZ(w_rms, A_eddy,
                                                 E_eddy, DOP, DOPM)
                        )/h_mode
        return DOPM_time_deriv

@numba.jit()
def AB3_time_step(array, delta_t, time_derivs):
    new_array = array + delta_t*(23.*time_derivs[0,:] - 16*time_derivs[1,:] + 5*time_derivs[2,:])/12.
    return new_array

@numba.jit()
def enforce_BCs(PO4, PO4M, DOP, DOPM, gyre_edge_po4, gyre_edge_DOP):
    # ege
    PO4[-1] = gyre_edge_po4
    # centre zero-gradient
    PO4[0] = PO4[1]
    PO4M[0] = PO4M[1]

    # edge
    DOP[-1] = gyre_edge_DOP
    # centre zero-gradient
    DOP[0] = DOP[1]
    DOPM[0] = DOPM[1]

    return PO4, PO4M, DOP, DOPM

@numba.jit()
def cycle_time_derivs(dPO4dt, dPO4Mdt, dDOPdt, dDOPMdt):

    dPO4dt[2,:] = dPO4dt[1,:]
    dPO4dt[1,:] = dPO4dt[0,:]
    dPO4Mdt[2,:] = dPO4Mdt[1,:]
    dPO4Mdt[1,:] = dPO4Mdt[0,:]

    dDOPdt[2,:] = dDOPdt[1,:]
    dDOPdt[1,:] = dDOPdt[0,:]
    dDOPMdt[2,:] = dDOPMdt[1,:]
    dDOPMdt[1,:] = dDOPMdt[0,:]

    return dPO4dt, dPO4Mdt, dDOPdt, dDOPMdt

def run_model_2L(PO4,PO4M,DOP,DOPM,params,numerics):
    dPO4dt = np.zeros((3,len(numerics['r'])))# + 1e-17
    dPO4Mdt = np.zeros((3,len(numerics['r'])))

    dDOPdt = np.zeros_like(dPO4dt)
    dDOPMdt = np.zeros_like(dPO4dt)
    
    # convert units in parameters to SI
    params['w_ek'] = params['w_ek']/(365.*86400.)
    params['w_rms'] = params['w_rms']/86400.
    params['alpha'] = params['alpha']/(365.*86400.)
    params['gamma_remin'] = params['gamma_remin']/(365.*86400.)
    
    for timestep in xrange(int(numerics['nsteps'])):
        
        dPO4_dr = d_dr(PO4, numerics['r'])
        dPO4rr_dr = d_dr(numerics['r']*numerics['r']*PO4, numerics['r'])
        
        dPO4dt[0,:] =  calc_dPO4_dt(PO4, PO4M, DOP, params['kappa_h'], params['kappa_z'], numerics['r'], 
                                    params['h_euph'], params['h_mode'], dPO4_dr, params['w_ek'], dPO4rr_dr,
                                    params['alpha'], params['lambda_po4'], params['gamma_remin'],
                                    params['w_rms'], params['A_eddy'], params['E_eddy'])
        
        dPO4M_dr = d_dr(PO4M,numerics['r'])
        dPO4Mrr_dr = d_dr(numerics['r']*numerics['r']*PO4M,numerics['r'])
        dPO4Mdt[0,:] = calc_dPO4M_dt(PO4, PO4M, DOPM, dPO4M_dr, dPO4Mrr_dr, params['kappa_h'], 
                                     params['kappa_z'], numerics['r'], 
                                    params['h_euph'], params['h_mode'], params['w_ek'],
                                    params['f_DOP'], params['alpha'], params['lambda_po4'],
                                    params['gamma_remin'], params['PO4abyss'],
                                    params['w_rms'], params['A_eddy'], params['E_eddy'])

        dDOP_dr = d_dr(DOP,numerics['r'])        
        dDOPrr_dr = d_dr(numerics['r']*numerics['r']*DOP,numerics['r'])
        
        dDOPdt[0,:] =  calc_dDOPdt(PO4, DOP, DOPM, dDOP_dr, dDOPrr_dr, params['kappa_h'], 
                                     params['kappa_z'], numerics['r'], 
                                    params['h_euph'], params['h_mode'], params['w_ek'],
                                    params['f_DOP'], params['alpha'], params['lambda_po4'],
                                    params['gamma_remin'], params['PO4abyss'],
                                    params['w_rms'], params['A_eddy'], params['E_eddy'])

        dDOPM_dr = d_dr(DOPM,numerics['r'])
        dDOPMrr_dr = d_dr(numerics['r']*numerics['r']*DOPM,numerics['r'])
        
        dDOPMdt[0,:] =  calc_DOPM_dt(PO4M, DOP, DOPM, dDOPM_dr, dDOPMrr_dr, params['kappa_h'], 
                                     params['kappa_z'], numerics['r'], 
                                    params['h_euph'], params['h_mode'], params['w_ek'],
                                    params['f_DOP'], params['alpha'], params['lambda_po4'],
                                    params['gamma_remin'], params['DOPabyss'],
                                    params['w_rms'], params['A_eddy'], params['E_eddy'])

        # step forward in time        
        PO4 = AB3_time_step(PO4, numerics['dt'], dPO4dt)
        PO4M = AB3_time_step(PO4M, numerics['dt'], dPO4Mdt)
        DOP = AB3_time_step(DOP, numerics['dt'], dDOPdt)
        DOPM = AB3_time_step(DOPM, numerics['dt'], dDOPMdt)

        # enforce BCs
        PO4, PO4M, DOP, DOPM = enforce_BCs(PO4, PO4M, DOP, DOPM,
                                           params['gyre_edge_conc'], params['gyre_edge_DOP'])

        # cycle time derivatives
        dPO4dt, dPO4Mdt, dDOPdt, dDOPMdt = cycle_time_derivs(dPO4dt, dPO4Mdt, dDOPdt, dDOPMdt)
        
    return PO4,PO4M,DOP,DOPM,dPO4dt,dPO4Mdt,dDOPdt,dDOPMdt

def default_params():
    params = {}

    params['kappa_h'] = 500.      # m^2/s - horiz diffusion
    params['kappa_z'] = 1e-5     # m^2/s - vertical diffusion
    params['w_ek'] = -30.        # m/y - Ekman pumping velocity + eddy cancellation velocity averaged over gyre in m/year
    params['w_rms'] = 1.0          # m/s - eddy upwelling rms velocity in m/day 
    params['A_eddy'] = 0.2             # dimensionless - fraction of ocean covered by eddies
    params['E_eddy'] = 0.4    # dimensionless efficiency of eddies
    params['alpha'] = 8.199360000000001e-4    # mol/m^3/year - max community productivity 
    params['lambda_po4'] = 5e-4          # mol/m^3 - half saturation value for phosphate
    params['gamma_remin'] = 2. # 1/year - remineralisation timescale for DOP
    params['f_DOP'] = 0.67                # dimensionless - proportion of biology that is immediately remineralised as DOP
    params['gyre_edge_conc'] = 0.17e-3      # mol/m^3 - concentration of phosphate at edge of gyre in euphotic zone
    params['gyre_edge_DOP'] = 0.06e-3      # mol/m^3 - concentration of DOP at edge of gyre in euphotic zone
    params['PO4abyss'] = 0.5e-3      # mol/m^3 - concentration of PO4 in the abyss
    params['DOPabyss'] = 0.01e-3      # mol/m^3 - concentration of DOP in the abyss
    params['R_g'] = 2000e3 # metres - radius of gyre
    params['h_euph'] = 100. # metres - depth of euphotic zone
    params['h_mode'] = 150. # metres

    return params

def default_numerics():
    params = default_params()

    numerics = {}
    numerics['dt'] = 1.5e4 # seconds
    numerics['nr'] = 200 # grid points in r
    numerics['nsteps'] = 4e6 # number of timesteps to compute

    numerics['r'] = np.linspace(-5375.0,params['R_g']+5375.0,numerics['nr']+1)
    # Add one grid point half a grid spacing to the negative of the centre of the gyre - this 
    # means that setting the two final grid points to be equal to each other ensures no gradients over the centre. 

    return numerics


