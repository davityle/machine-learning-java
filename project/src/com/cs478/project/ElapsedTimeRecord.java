package com.cs478.project;

import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 */
public class ElapsedTimeRecord implements CSV<ElapsedTimeRecord> {

    public final BaseRecord baseRecord;
    public final long duration;

    public ElapsedTimeRecord(BaseRecord baseRecord, long duration) {
        this.baseRecord = baseRecord;
        this.duration = duration;
    }

    @Override
    public String header() {
        return Stream.concat(baseRecord.fields(), fields()).collect(Collectors.joining(","));
    }

    @Override
    public Stream<String> fields() {
        return Stream.of("total_duration");
    }

    @Override
    public Stream<String> values() {
        return Stream.of(Long.toString(duration));
    }

    @Override
    public String toString() {
        return Stream.concat(baseRecord.fields(), fields()).collect(Collectors.joining(","));
    }

    @Override
    public int compareTo(ElapsedTimeRecord o) {
        return baseRecord.compareTo(o.baseRecord);
    }
}
