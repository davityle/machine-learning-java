package com.cs478.project;

import com.codepoetics.protonpack.StreamUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PreProcess {

    public static <T> List<T> mapAndCollect(Map<Long, List<BaseRecord>> customerRecords, Function<List<BaseRecord>, List<T>> mapFunc) {
        return customerRecords
                .keySet()
                .stream()
                .map(custId -> mapFunc.apply(customerRecords.get(custId)))
                .flatMap(Collection::stream)
                .collect(Collectors.toList());
    }

    public static void main(String[] args) throws IOException {

        BaseRecord[] records = Files.readAllLines(Paths.get("train.csv"))
                .stream()
                .filter(line -> !line.startsWith("customer_ID"))
                .map(line -> line.split(","))
                .map(BaseRecord::new)
                .toArray(BaseRecord[]::new);

        Map<Long, List<BaseRecord>> customerRecords = Arrays.stream(records)
                .collect(Collectors.groupingBy(record -> record.customer_ID));

        List<ElapsedTimeRecord> elapsedTimeRecords = mapAndCollect(customerRecords, (cust) -> {
            Collections.sort(cust);
            long timeDiff = ChronoUnit.MINUTES.between(cust.get(0).time, cust.get(cust.size() - 1).time);
            if (timeDiff < 0) {
                timeDiff = (24 * 60) - timeDiff;
            }
            final long finalDiff = timeDiff;
            return cust.stream().map(r -> new ElapsedTimeRecord(r, finalDiff)).collect(Collectors.toList());
        });

        List<IsContiguousRecord> contiguousRecords = mapAndCollect(customerRecords, (cust) -> {
            Collections.sort(cust);
            int i;
            for (i = 1; i < cust.size(); i++) {
                if (Math.abs(ChronoUnit.MINUTES.between(cust.get(i - 1).time, cust.get(i).time)) > IsContiguousRecord.CONTIGUOUS_TIME) {
                    break;
                }
            }
            final boolean isContiguous = i == cust.size();
            return cust.stream().map(r -> new IsContiguousRecord(r, isContiguous)).collect(Collectors.toList());
        });
        List<ProcessedRecord> s = StreamUtils.zip(elapsedTimeRecords.stream(), contiguousRecords.stream(), (er, cr) -> new ProcessedRecord(er.baseRecord, er, cr)).filter(r -> r.baseRecord.record_type == 1).collect(Collectors.toList());
        printRecords(s::stream, "time_records_only_end.csv");
    }

    public static void printRecords(Supplier<Stream<? extends CSV>> records, String name) throws IOException {
        StringBuilder dr = new StringBuilder(records.get().findFirst().get().header()).append('\n');
        records.get().sorted().forEach(r -> dr.append(r).append('\n'));
        Files.write(Paths.get(name), dr.toString().getBytes());
    }

}
