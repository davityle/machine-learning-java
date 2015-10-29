package com.cs478.project;

import java.util.stream.Stream;

/**
 *
 */
public interface CSV<T> extends Comparable<T>{
    String header();
    Stream<String> fields();
    Stream<String> values();
}
