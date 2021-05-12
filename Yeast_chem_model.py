"""
A simplified model of Glycolysis and the TCA cycle using Gillespie's next reaction method.
"""

__author__ = 'Ata Kalirad'

__version__ = '1.1'


import os
import pickle
from tqdm import tqdm 
from copy import *
from itertools import *

import numpy as np
import random as rnd
import pandas as pd

#  set random number generator seed
np.random.set_state(('MT19937', np.array([
    3691495208, 2881111814, 3977485953,  126022579, 1276930617,
     355103692, 3248493791, 3009808844,  612188080,  248004424,
    1489588601,  173474438, 4039752635, 2508845774, 2622234337,
    2700397831, 1811893199, 2190136060, 2315726008, 1162460778,
    2341168633,  236659960, 3175264097, 3400454537,  427729918,
    4066770621,  567157494, 4014767970, 2930740323,  378300123,
    2705662117, 3891078126, 1960649845, 3044656210,  882045208,
    1570375463, 2086686192,  407452463, 2030931525, 2734889467,
    3712254193, 3949803070,  764947052, 2833180084, 2612938943,
    3513858645, 1012338082, 1723965053,   40253333, 3097240011,
    3472905330,  563287754,  704858225,  610145833, 2824639775,
    3671030693,  225662685, 4093017874,  488496843, 3011853058,
    3141429748, 2892388748, 1752852512, 1097583623, 3335701968,
    2741138771, 2366687650, 2909722827, 3896701472, 2855844360,
      14740992,  126288255,  556395335, 3606698449, 1990092369,
    1892289888, 1025326265, 3335170268, 2955298765, 2086040311,
    2644433388, 1986237624,  831065590, 2567078834, 3535829239,
    1597256603,  781977323, 2945733169, 3479378352, 3652557111,
    1100223342,  235212556, 2599186570,  899620665,  675417868,
    1297279698, 3980368873, 1671894382, 3219957975,  129492647,
     369423255, 1887390651,  536695139, 3467326731,  577893063,
    3628585169, 2772043849,  369219244, 1271097627, 1346409244,
    2331891903,   39930497, 2068899034,  539572370, 4195007861,
    3495378688, 3377756157, 2835342219, 3699793011, 3321615441,
    2211559076, 2398792755, 2796307031,  818646352,  355446500,
    2946711801, 1049957619,  561188288, 2829760282,   55894884,
    1568204679, 1764468784, 1959965565, 4065967902, 3887804509,
    3833073650, 3717783102, 1837449653,  528963116, 4121548680,
    2402147957, 2202929313,  747086954, 3205182257, 1631864764,
     858833100,  148465241,   17458708, 2148761251, 3002919548,
    3773743659, 2611894356, 2275521209, 3027905006, 2234470309,
    2709870512, 1052969526, 3035329785,  110428213, 2893701759,
    2512125031, 3045322315, 2322452091, 3576747394, 2006737455,
     124047895, 3870223050, 3757797920,  698743578,  701653240,
    3561309206,   39541368, 2659965257, 3356207001,  698671102,
    1967130233, 3584965340, 3302789650,  104792115,  989737788,
    1289315250, 2742066874,  943135962, 2610987463, 4156696495,
    1957093316, 1880989243,  211024555, 1594171485, 2646518040,
    1391570537, 2982210346, 3225750783, 1452478140, 1063288625,
    2782363442,  333182057, 2864780704, 3890295634, 1022925971,
     226535384, 2132360150,   74977604, 4208008791, 1697651592,
    4029637378,  397828762, 2954491996, 1120498466, 3197759375,
    2646537589, 2903140119,  580234113, 2324229766, 1485090247,
    3173462698, 1441000100, 3212564317,  598271368, 1052134622,
    2751284206, 4040281713, 2630844601, 1921303308,  861775468,
    3522939180, 2855935558, 3227004083, 4121725263,  805407916,
    1207185676,  785322196, 3104463214, 3070205549, 1984686779,
       5199855, 2585264490, 3703002136, 3352578045,  257641487,
    1613285168, 3845545412, 2884412656, 3795140597, 2864082431,
    1708426814,  661272124, 3359489670, 2989690080, 1120054048,
    3029239860, 2037244341, 3411962036, 3468887812, 1294329307,
    1967939294, 1668712931, 1560596708, 2986374405, 3266952874,
    1758277657, 3876598642, 1149698899, 1548677880, 2464327872,
     466262570, 2573332645, 3577605405, 3511489634, 3001210402,
    4047160993, 1096981688, 1365437714,  967187969, 2651685599,
    4258218418,  618336653, 1813338507, 4161534170, 1206855048,
    3766692676, 1984622584, 1256641952, 2293866774, 2566572107,
    1296931689,  202959755, 3331103372, 3095866549, 1832670718,
    3588629070,  533366259,  301078755, 1299816886, 2612908898,
    1142385071, 4044229138,  392786907, 1473264101,  171872184,
    2873022820, 1878820461,   88690985, 3019565333, 2121461097,
    1522107992, 1733374438, 2311932879,  556408593, 1461835210,
    1423528436,  819211315,  889069790, 3086689727, 1730639543,
    1216615289, 2492159266, 1809961698, 1659780200, 3125102201,
    1711752707, 2723337471, 2521518355, 3884672928, 1313721188,
    1901655237, 3962083231,  757934816, 2196008247, 2111842931,
    2965600004, 1312840433, 3455017541,  545137641, 2279641585,
    2939005091, 1537081838, 2463922331, 1996015762, 1196027276,
     906621855, 1704400250,   76236737,  136244169,  619138087,
      98595120,  719278264, 1334390246, 3171154143, 1280182795,
    2215843496, 2676742417, 2197843524, 1396698993,  609335212,
     723295525, 3605167513, 4155694342, 3017089897, 1955520678,
    4067049686, 3239743094, 1221155545, 4095319239,  425400349,
    1806147353, 3671105575,  627163234, 1861707767,  274296576,
     638507216, 1649469686,  608691281, 4232809768,  611030651,
     853789168, 1733062866,  540453354,   11996619, 2695864391,
    2050310856,  141509199,  252149019, 3547463915,  329855083,
    2856249739, 3735981321, 2875626876, 2379144635,   13062386,
    1562227109, 1191505353, 3203340427, 2778675184, 2770557127,
    3644383877, 1790071106, 2240228460, 1676798968,  863141840,
    1175886689, 1178806726,  358678487, 3328835908, 2633561969,
    4074930335,  772447425, 3430950121, 3352113867,  701629620,
      25420967, 3791888554, 1412926413,  791735289,  161600651,
     506627594, 4220683170,  539553216,  176491711,  870303302,
    2405928427,  673609577,  616683903, 2009922698, 2088461621,
     631204850,  495792565, 1105952597, 1332646700,   23124919,
    2539330986, 1231655942, 1860851071, 3651186456, 2775290123,
    3681258608,  637100105, 4220549846, 3186875083, 3856908269,
    3867761132, 3985657986, 4173577631,  552539584, 2204479092,
    4165177831, 2396591349, 3474222162, 2920321345, 3906718099,
     515536637,  991766590, 2116510279,  482084635, 4005496942,
     374235227, 1711760850, 3750465691,  101652558, 3589303631,
    1360138030, 1382922742,  340163774, 2692240084, 2626346609,
    3041178492, 3616792294,  699158099, 1180482576, 3504356230,
    1897868877,  464615571, 3149754153, 2219112250, 2421357980,
    3182082688, 3145015709, 2579307737, 3490881071, 2970802492,
    3235037551, 1994684505,  355293861, 2682386071, 1408942224,
    3272168205, 3715571520,  476379336, 3644917929,  666542692,
    2680727545,  560661664, 1022989241,  806139402,  495605276,
     462775794, 2795097035, 1348129402, 4137368209, 2768709750,
    2129930451,  422284347, 1297682726, 1252742143, 3031031382,
      75134366, 3411139976, 1654986716,  532012083, 1253013106,
    1814002341,  584805750, 4151151859,  279516416, 2068669679,
    1452548111,  255585988, 2731877417,  805942443, 3209104026,
    1105115396, 1929339947, 3829736722, 2980275336, 2169476831,
     784792828, 3572862771, 1057808935, 1774004947, 3086076921,
     969435958, 4291618669,  892653473, 2713995907, 2137887400,
    2565641007, 1417836736,  415508859, 1624683723,   23763112,
     518111653, 2355447857, 2023934715,  934168085, 2250448450,
     450387908, 1069332538, 4170085337, 2145735300, 2298032455,
    1437026749, 2863147795, 3273446986, 1979692197, 3208629490,
    2080357079,  584771674, 1802076639, 2018580439, 4261231470,
    1708636029, 3602321445,   18885205, 1940272685, 4187271341,
    1647123067, 1450487947, 3463781280, 3759557524,  493883757,
    3901885447, 3190687437,  742916954, 3176758487, 3010187255,
     936898923, 1805555016, 1981187811, 1196213096, 3067885662,
    2550095824, 3396199635, 3614915928, 1977375679, 2173583078,
    2643789240, 2587955166, 2158941995, 2347766906, 1711205114,
      66633020, 3977356823, 1510661526, 3048960083,   51672689,
    3596587592, 4038438382, 4019922490, 2146383929, 1692948176,
    1233895739, 3938222851, 2698966080, 2950467396, 1878048591,
    3547155317, 3627364723,  906814924, 1075129814, 3302437944,
    2756803960, 2719380291, 1774084191, 2789415893, 4095723844,
    1297221824, 1938199324, 4112704123, 1741415251, 1105144176,
    1259977468,  131064353, 4036118418,  311279014], dtype=np.uint32),
    624, 0, 0.0))

class Species(object):
    """Chemical species and its corresponding attributes.
    """

    def __init__(self, init_n, regulator=False, energetic=False):
        """Initialize the Species object.
        
        Arguments:
            init_n {int} -- The intital number of the chemical species.
        
        Keyword Arguments:
            regulator {bool} -- Indicate if the species simply affect the 
                                reaction rate and is not consumed 
                                (default: {False})

            energetic {bool} -- Indicate if the species is an energy carrier 
                                -e.g., ATP (default: {False})
        """
        self.n = init_n
        self.regulator = regulator
        self.energetic = energetic

    def increase(self):
        self.n += 1
        
    def decrease(self):
        if self.n !=0:
            self.n -= 1

class Yeast(object):
    
    def __init__(self, GLC, PEP, PYR, ETHANOL, PYK1_in, PYK1_ac, PYK2_in, PYK2_ac, k_lst, regulation=True, wt=True):
        """[summary]
        
        Arguments:
            GLC {int} -- the initial number of this species in a yeast.
            PEP {int} -- the initial number of this species in a yeast.
            PYR {int} -- the initial number of this species in a yeast.
            ETHANOL {int} -- the initial number of this species in a yeast.
            PYK1_in {int} -- the initial number of this species in a yeast.
            PYK1_ac {int} -- the initial number of this species in a yeast.
            PYK2_in {int} -- the initial number of this species in a yeast.
            PYK2_ac {int} -- the initial number of this species in a yeast.
            k_lst {list} -- The list of mesoscopic reaction rates for each reaction.
        
        Keyword Arguments:
            regulation {bool} -- Indicate if the yeast utilizes the regulatory network (default: {True})
            wt {bool} -- Indicate if the yeast is not mutated (default: {True})
        """

        assert len(k_lst) == 8
        self.n_reactions = len(k_lst)
        self.regulation = regulation 
        self.wild_type = wt
        self.k = k_lst
        self.clock = 0
        self.init_GLC_count = GLC
        self.GLC = Species(GLC)
        self.PEP = Species(PEP)
        self.PYR = Species(PYR)
        self.ETHANOL = Species(ETHANOL) 
        self.ATP = Species(0., energetic=True) 
        self.ATP_r = Species(0., energetic=True) 
        self.PYK1_in = Species(PYK1_in, regulator=True) 
        self.PYK1_ac = Species(PYK1_ac, regulator=True) 
        self.PYK2_in = Species(PYK2_in, regulator=True) 
        self.PYK2_ac = Species(PYK2_ac, regulator=True) 
        self.sink = Species(0.)
        self.init_reactions()
        self.init_history()

    @property 
    def num_mol(self):
        """Get the total number of chemical species 

        Returns:
            int : total num molecules
        """
        tot_n =  self.GLC.n + self.PEP.n + self.PYK1_in.n + self.PYK1_ac.n \
            + self.PYK2_in.n + self.PYK2_ac.n + self.PYR.n + self.ATP.n \
            + self.ETHANOL.n + self.ATP_r.n
        return tot_n 


    def set_strat_prop(self, frm_prop):
        """Set the proportion fermenting cells.

        Args:
            frm_prop (int): Number of fermenting cells (used by Population class)
        """
        self.strat_prop = frm_prop

    def set_strategy(self, strategy):
        """Set the strategy of the cell (resp or ferm)
        """
        self.strategy = strategy

    def init_reactions(self):
        """Initialize dictionaries which contan the reactions of the system 
        """
        self.reactants = {0: [self.GLC],\
                          1: [self.PEP],\
                          2: [self.PYR, self.PYK1_ac],\
                          3:[self.PYK2_ac, self.PYR],\
                          4:[self.PYK1_in] ,\
                          5:[self.PYK2_in],\
                          6:[self.PYK1_ac],\
                          7:[self.PYK2_ac]}
        
        self.products = {0: [self.PEP, self.PEP], \
                         1: [self.PYR, self.ATP], \
                         2:[self.ETHANOL],\
                         3:[self.ATP_r], \
                         4:[self.PYK1_ac],\
                         5:[self.PYK2_ac],  \
                         6:[self.PYK1_in] , \
                         7:[self.PYK2_in]}

    def init_history(self):
        """Initialize history
        """
        self.GLC_t = [] 
        self.PEP_t = [] 
        self.PYR_t = [] 
        self.ETHANOL_t = [] 
        self.ATP_t = [] 
        self.ATP_r_t = [] 
        self.PYK1_ac_t = []
        self.PYK2_ac_t = []
        self.PYK1_in_t = []
        self.PYK2_in_t = []
        self.time = []
        
    def update_history(self):
        """Update history
        """
        self.GLC_t.append(self.GLC.n) 
        self.PEP_t.append(self.PEP.n) 
        self.PYR_t.append(self.PYR.n) 
        self.ETHANOL_t.append(self.ETHANOL.n) 
        self.ATP_t.append(self.ATP.n)
        self.ATP_r_t.append(self.ATP_r.n) 
        self.PYK1_ac_t.append(self.PYK1_ac.n)
        self.PYK2_ac_t.append(self.PYK2_ac.n)
        self.PYK1_in_t.append(self.PYK1_in.n)
        self.PYK2_in_t.append(self.PYK2_in.n)
        self.time.append(self.clock)

    @property
    def stats(self):
        """Generate a stats dictionary 
        """
        stats = {}
        stats['GLC_t'] = self.GLC_t 
        stats['PEP_t'] = self.PEP_t 
        stats['PYR_t'] = self.PYR_t 
        stats['ETHANOL_t'] = self.ETHANOL_t  
        stats['ATP_t'] = self.ATP_t  
        stats['ATP_r_t'] = self.ATP_r_t  
        stats['PYK1_ac_t'] = self.PYK1_ac_t
        stats['PYK2_ac_t'] = self.PYK2_ac_t
        stats['time'] = self.time
        return stats

    def get_propensity(self, ind, reactants, k):
        """[summary]

        Args:
            ind (int): The index of the reaction
            reactants (list): a list of reactants 
            k (float): the mesoscopic reaction constant 

        Returns:
            float: The propensity of the reaction
        """
        n = self.num_mol
        if self.regulation == True:
            if ind == 4: #PYK1 activation
                a = 1
                b = 0.5
                c = 4
                prop = (a* (self.GLC.n/n)**c)/(b**c + (self.GLC.n/n)**c) * reactants[0].n
            elif ind == 5: #PYK2 activation
                prop =  1e-1* reactants[0].n
            elif ind == 6: #PYK1 inactivation
                a = 5
                b = 0.5
                c = 2
                prop = (a* (self.PEP.n/n)**c)/(b**c + (self.PEP.n/n)**c) * reactants[0].n  
            elif ind == 7: #PYK2 inactivation
                a = 1
                b = 0.5
                c = 6
                prop = (a * (self.GLC.n/n)**c)/(b**c + (self.GLC.n/n)**c) * reactants[0].n
            else:
                prop = k
                for i in reactants:
                    prop *= i.n
        else:
             #PYK1 activation
            if ind == 4:
                if self.strategy == 'ferm':
                    prop = 1. * reactants[0].n
                else:
                    prop = 0.
            #PYK2 activation
            elif ind == 5: 
                if self.strategy == 'resp':
                    prop = 1. * reactants[0].n
                else:
                    prop = 0.
            else:
                prop = k
                for i in reactants:
                    prop *= i.n
        return prop
    
    def first_reaction_method(self):
        """Execute Gillrdpie's first reaction method

        Returns:
            tuple: The reaction and its index
        """
        #step1: calculate propensities.
        props = np.zeros(self.n_reactions)
        for i in range(len(props)):
            props[i] = self.get_propensity(i, self.reactants[i], self.k[i])
        #step2: get the reaction times
        times = []
        for i in props:
            if i > 0:
                times.append(1./i*(np.log(1./(rnd.random()))))
            else:
                times.append(np.inf)
        #step3: get smallest time and the index of the reaction with the smallest time
        min_time = np.min(times)
        min_time_ind = np.argmin(times)
        return min_time, min_time_ind

    def execute_reaction(self, min_time, min_time_ind):
        """Execute the reaction picked by the first reaction method

        Args:
            min_time (float) 
            min_time_ind (float)
        """
        if min_time == np.inf:
            self.clock += 0
        else:
            for j in self.products[min_time_ind]:
                if min_time_ind == 3:
                    for m in range(17):
                        j.increase()
                else:
                    j.increase()
            for i in self.reactants[min_time_ind]:
                if (min_time_ind == 2 or min_time_ind == 3) and i.regulator:
                    pass
                else:
                    i.decrease() 
            self.clock += min_time
        
    def simulate(self, T, fixed_source=True, verbose=False):
        """Simulate the chemical reactions in a single yeast cell

        Args:
            T (int):  The number of steps in simulation
            fixed_source (bool, optional): Indicate if the glucose remins constant in 
                                   the environment. Defaults to True.
            verbose (bool, optional): Display a progress bar during the simulation. Defaults to False.
        """
        self.update_history()
        if verbose:
            for i in tqdm(range(T)):
                a, b = self.first_reaction_method()
                self.execute_reaction(a, b)
                if fixed_source:
                    self.GLC.n = self.init_GLC_count
                self.update_history()
        else:
            for i in range(T):
                a, b = self.first_reaction_method()
                self.execute_reaction(a, b)
                if fixed_source:
                    self.GLC.n = self.init_GLC_count
                self.update_history()

class Population(object):

    def __init__(self, cell, N, fixed_source=True, duplication=False):
        """Initialize Population object.
        
        Arguments:
            cell {Yeast} -- The initial yeast cell.
            N {int} -- The number of cells.
        
        Keyword Arguments:
            fixed_source {bool} -- If True the amount of glucose remains 
                                   fixed during simulation (default: {True})
            duplication {bool} -- Indicates if yeast cells would undergo 
                                  division after accumulating enough 
                                  biomass (default: {False})
        """
        self.init_cell = cell
        self.fixed_source = fixed_source
        self.duplication = duplication
        self.glc_pool = self.init_cell.GLC.n
        self.pop = []
        for i in range(N):
            cell = deepcopy(cell)
            if self.init_cell.regulation == False:
                if len(self.pop) < self.init_cell.strat_prop:
                    cell.set_strategy('ferm')
                else:
                    cell.set_strategy('resp')
            self.pop.append(cell)
        self.clock = 0
        self.init_history()
        self.update_history()

    @property
    def glc_pool(self):
        return self._glc_pool

    @glc_pool.setter
    def glc_pool(self, v):
        self._glc_pool = v

    @property
    def N(self):
        """Population size

        Returns
        -------
        int
            N
        """
        return len(self.pop)

    def change_pop(self, new_pop):
        self.pop = deepcopy(new_pop)

    def update_time(self, t):
        self.clock += t

    def init_history(self):
        self.GLC_t_ind = []
        self.ATP_r_t_ind = []
        self.ETHANOL_t_ind = []
        self.GLC_pool_t = []
        self.pop_size = []
        self.num_mut = []
        self.times = []

    def update_history(self):
        self.GLC_t_ind.append([i.GLC.n for i in self.pop])
        self.ETHANOL_t_ind.append([i.ETHANOL.n for i in self.pop])
        self.ATP_r_t_ind.append([i.ATP_r.n for i in self.pop])
        self.GLC_pool_t.append(self.glc_pool)
        self.pop_size.append(self.N)
        num_mut = self.N - np.sum([i.wild_type for i in self.pop])
        self.num_mut.append(num_mut)
        self.times.append(self.clock)

    @property
    def stats(self):
        stats = {}
        stats['Glc_ind'] = pd.DataFrame(self.GLC_t_ind)
        stats['EtOH_ind'] = pd.DataFrame(self.ETHANOL_t_ind)
        stats['ATP_r_ind'] = pd.DataFrame(self.ATP_r_t_ind)
        stats['Glc'] = np.mean(stats['Glc_ind'], axis=1)
        stats['EtOH'] = np.mean(stats['EtOH_ind'], axis=1)
        stats['ATP_r'] = np.mean(stats['ATP_r_ind'], axis=1)
        stats['Glc_std'] = np.std(stats['Glc_ind'], axis=1)
        stats['EtOH_std'] = np.std(stats['EtOH_ind'], axis=1)
        stats['ATP_r_std'] = np.std(stats['ATP_r_ind'], axis=1)
        stats['times'] = self.times
        return stats
        
    def simulate_step(self):
        """Simulate the reactions in population of yeasts by executing the reaction with the minimum time in the population.
        """
        times = np.zeros(self.N)
        times_ind = np.zeros(self.N)
        for i in range(self.N):
            t, t_ind = self.pop[i].first_reaction_method()
            times[i] = t
            times_ind[i] = t_ind
        sorted_times = np.argsort(times)
        for i in sorted_times:
            glc_i_before = self.pop[i].GLC.n 
            self.pop[i].execute_reaction(times[i], times_ind[i])
            glc_i_after = self.pop[i].GLC.n 
            diff = glc_i_after - glc_i_before
            if not self.fixed_source:
                new_pool = self.glc_pool + diff
                if new_pool < 0:
                    new_pool = 0
                self.glc_pool = new_pool
                if diff != 0:
                    for j in sorted_times[i:]:
                        self.pop[j].GLC.n = self.glc_pool
            else:
                for j in sorted_times[i:]:
                    self.pop[j].GLC.n = self.glc_pool
            self.pop[i].update_history()
        self.update_time(sorted_times[-1])
        self.update_history()

    def simulate_pop(self, T, verbose=False):
        while self.clock < T:
            self.simulate_step()
            if verbose:
                if not i%1000:
                    print(i),

if __name__ == "__main__":
    import doctest
    doctest.testmod()

        
             
            

    

