import csv
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m', action='store', default='4096',type=int)
parser.add_argument('--n', action='store', default='4096',type=int)
parser.add_argument('--k', action='store', default='4096',type=int)
parser.add_argument('--csv', action='store', default='log_imma.csv')
parser.add_argument('--out', action='store', default='out_imma.txt')

args = parser.parse_args()

csv_file = args.csv
out_file = args.out
m=args.m
n=args.n
k=args.k
fout = open(out_file, "w")
fout.write("m="+str(m)+", n="+str(n)+", k="+str(k)+"\n")
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if len(row)>4:
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
                fout.write("perf scaling factor: "+str(sf)+"\n")
                fout.write("kernel name: " + row[4])
                fout.write("\n=======================================================================================\n")
                peakperfpercycle = row[PeakWorkPerCycle_index]
                fout.write("peak perf per cycle: "+peakperfpercycle.replace(",","") )
                fout.write("\n=======================================================================================\n")
                peakworkcyclespersecond = row[CyclePerSecond_index]
                fout.write("peak work Giga cycle per second: "+str(float(peakworkcyclespersecond.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                peak_perf = float(peakperfpercycle.replace(",","")) * float(peakworkcyclespersecond.replace(",",""))*sf
                fout.write("peak performance TFLOPS: "+str(peak_perf*1e-12))
                fout.write("\n=======================================================================================\n")
                peakdrambytes = row[PeakDRAMPerCycle_index]
                fout.write("peak dram bytes per cycle: "+peakdrambytes.replace(",",""))
                fout.write("\n=======================================================================================\n")
                peakdramcyclespersecond = row[DRAMCyclesPerSecond_index]
                fout.write("peak dram Giga cycles per second: "+ str(float(peakdramcyclespersecond.replace(",",""))*1e-9)) 
                fout.write("\n=======================================================================================\n")

                peak_BW = float(peakdrambytes.replace(",","")) * float(peakdramcyclespersecond.replace(",",""))
                fout.write("peak dram BW GB/s: " + str(peak_BW*1e-9))
                fout.write("\n=======================================================================================\n")
                achievedworkpercycle = row[AchievedWorkPerCyclesIMMA_index]
                fout.write("Achieved work per TC per cycle: "+ achievedworkpercycle.replace(",",""))
                fout.write("\n=======================================================================================\n")
                achievedcyclespersecond = row[AchievedCyclesPerSecond_index]
                fout.write("Achieved Giga cycles per second: "+ str(float(achievedcyclespersecond.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                achievedtraffic = row[AchievedDRAMTraffic_index]
                fout.write("DRAM level achieved traffic (data rate in Giga byte per second) "+ str(float(achievedtraffic.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                kernel_perf = float(achievedworkpercycle.replace(",","")) * float(achievedcyclespersecond.replace(",",""))*sf
                fout.write("kernel performance (TFLOPS): "+ str(kernel_perf*1e-12))
                fout.write("\n=======================================================================================\n")
                ridge_point = peak_perf / peak_BW
                fout.write("DRAM level ridge point : "+ str(ridge_point))
                fout.write("\n=======================================================================================\n")
                arithmetic_intensity = kernel_perf / float(achievedtraffic.replace(",",""))
                fout.write("DRAM level arithmetic intensity (Op/Byte): "+ str(arithmetic_intensity))
                fout.write("\n=======================================================================================\n")
                peakl2bytes = row[PeakL2PerCycle_index]
                fout.write("peak l2 bytes per cycle: "+peakl2bytes.replace(",",""))
                fout.write("\n=======================================================================================\n")
                peakl2cyclespersecond = row[L2CyclesPerSecond_index]
                fout.write("peak l2 Giga cycles per second: "+ str(float(peakl2cyclespersecond.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                peakl2_BW =  float(peakl2bytes.replace(",","")) * float(peakl2cyclespersecond.replace(",",""))
                fout.write("peak l2 BW GB/s: " + str(peakl2_BW*1e-9))
                fout.write("\n=======================================================================================\n")
                achievedl2traffic = row[AchievedL2Traffic_index]
                fout.write("l2 level achieved traffic (data rate in Giga byte per second) "+ str(float(achievedl2traffic.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                ridge_point_l2 = peak_perf / peakl2_BW
                fout.write("l2 level ridge point : "+ str(ridge_point_l2))
                fout.write("\n=======================================================================================\n")
                arithmetic_intensity_l2 = kernel_perf / float(achievedl2traffic.replace(",",""))
                fout.write("l2 level arithmetic intensity (Op/Byte): "+ str(arithmetic_intensity_l2))
                fout.write("\n=======================================================================================\n")
                peakl1bytes = row[PeakL1PerCycle_index]	
                fout.write("peak l1 bytes per cycle: "+peakl1bytes.replace(",",""))
                fout.write("\n=======================================================================================\n")
                peakl1cyclespersecond = row[L1CyclesPerSecond_index]
                fout.write("peak l1cycles Giga per second: "+ str(float(peakl1cyclespersecond.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                #peakl1_BW =  float(peakl1bytes.replace(",","")) * float(peakl1cyclespersecond.replace(",",""))
                fout.write("peak l1 BW GB/s: " + str(peakl1_BW*1e-9))
                fout.write("\n=======================================================================================\n")
                achievedl1traffic = row[AchievedL1Traffic_index]
                fout.write("l1 level achieved traffic (data rate in Giga byte per second) "+ str(float(achievedl1traffic.replace(",",""))*1e-9))
                fout.write("\n=======================================================================================\n")
                ridge_point_l1 = peak_perf / peakl1_BW
                fout.write("l1 level ridge point : "+ str(ridge_point_l1))
                fout.write("\n=======================================================================================\n")
                arithmetic_intensity_l1 = kernel_perf / float(achievedl1traffic.replace(",",""))
                fout.write("l1 level arithmetic intensity (Op/Byte): "+ str(arithmetic_intensity_l1))
                fout.write("\n=======================================================================================\n")
fout.close()	


