package com.cs478.project;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 */
public class IsContiguousRecord implements CSV<IsContiguousRecord> {

    public static final int CONTIGUOUS_TIME = 15;

    public final BaseRecord baseRecord;
    public final long is_contiguous;

    public IsContiguousRecord(BaseRecord baseRecord, boolean is_contiguous) {
        this.baseRecord = baseRecord;
        this.is_contiguous = is_contiguous ? 1 : 0;
    }

    @Override
    public String header() {
        return Stream.concat(baseRecord.fields(), fields()).collect(Collectors.joining(","));
    }

    @Override
    public Stream<String> fields() {
        return Stream.of("is_contiguous");
    }

    @Override
    public Stream<String> values() {
        return Stream.of(Long.toString(is_contiguous));
    }

    @Override
    public String toString() {
        return Stream.concat(baseRecord.values(), values()).collect(Collectors.joining(","));
    }

    @Override
    public int compareTo(IsContiguousRecord o) {
        return baseRecord.compareTo(o.baseRecord);
    }
}
