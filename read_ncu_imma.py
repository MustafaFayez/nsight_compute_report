import csv
import pdb
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--m', action='store', default='4096',type=int)
parser.add_argument('--n', action='store', default='4096',type=int)
parser.add_argument('--k', action='store', default='4096',type=int)
parser.add_argument('--csv', action='store', default='log_imma.csv')
parser.add_argument('--log', action='store', default='out_imma.log')
parser.add_argument('--out', action='store', default='out_imma.csv')

args = parser.parse_args()

csv_file = args.csv
log_file = args.log
out_file = args.out
m=args.m
n=args.n
k=args.k
flog = open(log_file, "w")
flog.write("m="+str(m)+", n="+str(n)+", k="+str(k)+"\n")

if Path(out_file).is_file():
    fout = open(out_file, "a")
else:
    fout = open(out_file, "w")
    fout.write("m,n,k,mem-bound-l1,mem-bound-l2,mem-bound-dram,op-byte-l1,op-byte-l2,op-byte-dram,bw-l1,bw-l2,bw-dram,ridge-l1,ridge-l2,ridge-dram\n")
 
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if len(row)>26:
            #import pdb; pdb.set_trace()
            if row[0].find('ID') > -1:
                PeakWorkPerCycle_index = row.index('sm__inst_executed_pipe_tensor_op_imma.sum.peak_sustained')
                CyclePerSecond_index = row.index('sm__cycles_elapsed.avg.per_second')
                PeakDRAMPerCycle_index = row.index('dram__bytes.sum.peak_sustained')
                DRAMCyclesPerSecond_index = row.index('dram__cycles_elapsed.avg.per_second')
                PeakL2PerCycle_index = row.index('lts__t_bytes.sum.peak_sustained')
                L2CyclesPerSecond_index = row.index('lts__cycles_elapsed.avg.per_second')
                PeakL1PerCycle_index = row.index('l1tex__t_bytes.sum.peak_sustained')
                L1CyclesPerSecond_index = row.index('l1tex__cycles_elapsed.avg.per_second')
                AchievedWorkPerCycles_index= row.index('smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed')
                AchievedWorkPerCyclesIMMA_index= row.index('smsp__inst_executed_pipe_tensor_op_imma.sum.per_cycle_elapsed')
                AchievedCyclesPerSecond_index = row.index('smsp__cycles_elapsed.avg.per_second')
                AchievedDRAMTraffic_index = row.index('dram__bytes.sum.per_second')
                AchievedL2Traffic_index = row.index('lts__t_bytes.sum.per_second')
                AchievedL1Traffic_index = row.index('l1tex__t_bytes.sum.per_second')

            if row[4].find('gemm') > -1:
                # based on GPU model and operation datatype we decide the performance factor based on this issue in nsight compute:
                # https://forums.developer.nvidia.com/t/why-the-compute-throughputs-value-is-different-from-the-actual-performance-peak-performance/227563
                # from the datasheet I found these factors.
                # also the L1 BW calc was wrong as per this issue https://forums.developer.nvidia.com/t/confused-about-the-l1-smem-bw-reported-by-nsight-compute-hierarchical-roofline-plots/232524/3
                # so i took the fixed numbers from datasheets
                if row[4].find('turing') > -1:
                    print("turing")
                    sf = 512
                    peakl1_BW = 14*1024*1024*1024*1024
                elif row[4].find('ampere') > -1 or row[4].find('sm80') > -1 or row[4].find('cutlass_80') > -1:
                    peakl1_BW =  19*1024*1024*1024*1024
                    if row[4].find('int8') > -1 or row[4].find('imma') > -1 or row[4].find('i16832gemm') > -1:
                        sf = 4096
                    else:
                        sf = 2048
                else:
                    peakl1_BW = 14*1024*1024*1024*1024
                    sf = 512
                flog.write("perf scaling factor: "+str(sf)+"\n")
                flog.write("kernel name: " + row[4])
                flog.write("\n=======================================================================================\n")
                peakperfpercycle = row[PeakWorkPerCycle_index]
                flog.write("peak perf per cycle: "+peakperfpercycle.replace(",","") )
                flog.write("\n=======================================================================================\n")
                peakworkcyclespersecond = row[CyclePerSecond_index]
                flog.write("peak work Giga cycle per second: "+str(float(peakworkcyclespersecond.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                peak_perf = float(peakperfpercycle.replace(",","")) * float(peakworkcyclespersecond.replace(",",""))*sf
                flog.write("peak performance TFLOPS: "+str(peak_perf*1e-12))
                flog.write("\n=======================================================================================\n")
                peakdrambytes = row[PeakDRAMPerCycle_index]
                flog.write("peak dram bytes per cycle: "+peakdrambytes.replace(",",""))
                flog.write("\n=======================================================================================\n")
                peakdramcyclespersecond = row[DRAMCyclesPerSecond_index]
                flog.write("peak dram Giga cycles per second: "+ str(float(peakdramcyclespersecond.replace(",",""))*1e-9)) 
                flog.write("\n=======================================================================================\n")

                peak_BW = float(peakdrambytes.replace(",","")) * float(peakdramcyclespersecond.replace(",",""))
                flog.write("peak dram BW GB/s: " + str(peak_BW*1e-9))
                flog.write("\n=======================================================================================\n")
                achievedworkpercycle = row[AchievedWorkPerCyclesIMMA_index]
                flog.write("Achieved work per TC per cycle: "+ achievedworkpercycle.replace(",",""))
                flog.write("\n=======================================================================================\n")
                achievedcyclespersecond = row[AchievedCyclesPerSecond_index]
                flog.write("Achieved Giga cycles per second: "+ str(float(achievedcyclespersecond.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                achievedtraffic = row[AchievedDRAMTraffic_index]
                flog.write("DRAM level achieved traffic (data rate in Giga byte per second) "+ str(float(achievedtraffic.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                kernel_perf = float(achievedworkpercycle.replace(",","")) * float(achievedcyclespersecond.replace(",",""))*sf
                flog.write("kernel performance (TFLOPS): "+ str(kernel_perf*1e-12))
                flog.write("\n=======================================================================================\n")
                ridge_point = peak_perf / peak_BW
                flog.write("DRAM level ridge point : "+ str(ridge_point))
                flog.write("\n=======================================================================================\n")
                arithmetic_intensity = kernel_perf / float(achievedtraffic.replace(",",""))
                flog.write("DRAM level arithmetic intensity (Op/Byte): "+ str(arithmetic_intensity))
                flog.write("\n=======================================================================================\n")
                peakl2bytes = row[PeakL2PerCycle_index]
                flog.write("peak l2 bytes per cycle: "+peakl2bytes.replace(",",""))
                flog.write("\n=======================================================================================\n")
                peakl2cyclespersecond = row[L2CyclesPerSecond_index]
                flog.write("peak l2 Giga cycles per second: "+ str(float(peakl2cyclespersecond.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                peakl2_BW =  float(peakl2bytes.replace(",","")) * float(peakl2cyclespersecond.replace(",",""))
                flog.write("peak l2 BW GB/s: " + str(peakl2_BW*1e-9))
                flog.write("\n=======================================================================================\n")
                achievedl2traffic = row[AchievedL2Traffic_index]
                flog.write("l2 level achieved traffic (data rate in Giga byte per second) "+ str(float(achievedl2traffic.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                ridge_point_l2 = peak_perf / peakl2_BW
                flog.write("l2 level ridge point : "+ str(ridge_point_l2))
                flog.write("\n=======================================================================================\n")
                arithmetic_intensity_l2 = kernel_perf / float(achievedl2traffic.replace(",",""))
                flog.write("l2 level arithmetic intensity (Op/Byte): "+ str(arithmetic_intensity_l2))
                flog.write("\n=======================================================================================\n")
                peakl1bytes = row[PeakL1PerCycle_index]	
                flog.write("peak l1 bytes per cycle: "+peakl1bytes.replace(",",""))
                flog.write("\n=======================================================================================\n")
                peakl1cyclespersecond = row[L1CyclesPerSecond_index]
                flog.write("peak l1cycles Giga per second: "+ str(float(peakl1cyclespersecond.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                #peakl1_BW =  float(peakl1bytes.replace(",","")) * float(peakl1cyclespersecond.replace(",",""))
                flog.write("peak l1 BW GB/s: " + str(peakl1_BW*1e-9))
                flog.write("\n=======================================================================================\n")
                achievedl1traffic = row[AchievedL1Traffic_index]
                flog.write("l1 level achieved traffic (data rate in Giga byte per second) "+ str(float(achievedl1traffic.replace(",",""))*1e-9))
                flog.write("\n=======================================================================================\n")
                ridge_point_l1 = peak_perf / peakl1_BW
                flog.write("l1 level ridge point : "+ str(ridge_point_l1))
                flog.write("\n=======================================================================================\n")
                arithmetic_intensity_l1 = kernel_perf / float(achievedl1traffic.replace(",",""))
                flog.write("l1 level arithmetic intensity (Op/Byte): "+ str(arithmetic_intensity_l1))
                flog.write("\n=======================================================================================\n")

                fout.write(str(m) + ", " + str(n) + ", " + str(k) + ", " + str(arithmetic_intensity_l1 < ridge_point_l1) + ", " \
                           + str(arithmetic_intensity_l2 < ridge_point_l2) + ", " + str(arithmetic_intensity < ridge_point) + ", "\
                        #   + str(arithmetic_intensity_l1) + "," + str(arithmetic_intensity_l2) + "," + str(arithmetic_intensity) + ","\
                          + "{:.2f}, {:.2f}, {:.2f}".format(arithmetic_intensity_l1, arithmetic_intensity_l2, arithmetic_intensity) + ", "\
                        #   + str(peakl1_BW) + "," + str(peakl2_BW) + "," + str(peak_BW) + "," \
                          + "{:.2f}, {:.2f}, {:.2f}".format(peakl1_BW*1e-9, peakl2_BW*1e-9, peak_BW*1e-9) + ", " \
                        #   + str(ridge_point_l1) + "," + str(ridge_point_l2) + "," + str(ridge_point) + "\n" )
                          + "{:.2f}, {:.2f}, {:.2f}".format(ridge_point_l1, ridge_point_l2, ridge_point) + "\n")
flog = open(log_file, "w")
flog.close()	


